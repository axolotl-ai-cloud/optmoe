# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""CPU harness for TorchDynamo graph breaks and recompiles in the OPTMoE Python layer.

Run as a script (one precision path per process, so the stubbed modules, the gloo group and
dynamo's caches never leak into another test):

    PYTHONPATH=src python tests/compile_harness.py bf16 --json

``deep_gemm``, the persistent backwards, the requant kernels and (when missing) ``triton``
are replaced by CPU stand-ins, so the custom ops' real Python impls run and return correctly
shaped tensors. What is traced is the real public API (``OptMoERuntime.forward`` /
``forward_lora``) inside a small two-layer MoE model, forward and backward, over steps
with varying routing and token counts.
"""

import argparse
import importlib.abc
import importlib.machinery
import json
import logging
import os
import sys
import types
from contextlib import contextmanager

import torch
import torch.nn.functional as F

PATHS = (
    "bf16",
    "mxfp8",
    "fp8",
    "bf16_lora",
    "mxfp8_lora",
    "fp8_lora",
    "nvfp4_lora",
    "dispatch",
    "dispatch_compact",
    "dispatch_bounded",
)
DISPATCH = ("dispatch", "dispatch_compact", "dispatch_bounded")
ARCH = {
    "bf16": 10,
    "mxfp8": 10,
    "fp8": 9,
    "bf16_lora": 10,
    "mxfp8_lora": 10,
    "fp8_lora": 9,
    "nvfp4_lora": 10,
    "dispatch": 10,
    "dispatch_compact": 10,
    "dispatch_bounded": 10,
}
EXPERTS, TOPK, HIDDEN, INTER, MAX_TOKENS, RANK = 4, 2, 256, 256, 64, 4
STEP_TOKENS = (48, 48, 48, 40, 56, 32)
# CPU-reachable adapter-gradient and forms routes; they only pick code inside the op impls
HARNESS_ENV = {
    "OPTMOE_LORA_WGRAD": "loop",
    "OPTMOE_LORA_MXFP8_POOLS": "bf16",
    "OPTMOE_LORA_MXFP8_FORMS": "merge",
    "OPTMOE_LORA_SM90_WGRAD": "project",
    "OPTMOE_LORA_MERGE": "torch",
}


class _Stub:
    def __call__(self, *args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return self

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return self

    def __getitem__(self, item):
        return self


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Stub()


class _TritonFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name == "triton" or name.startswith("triton."):
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        module = _StubModule(spec.name)
        module.__path__ = []
        return module

    def exec_module(self, module):
        pass


def stub_triton():
    """Import-only stand-in when Triton is not installed (macOS); no kernel is launched."""
    try:
        import triton  # noqa: F401
    except ImportError:
        # torch probes for triton lazily; settle those probes before the stand-in exists
        import torch._dynamo
        import torch._functorch._aot_autograd.graph_compile
        import torch._inductor.compile_fx
        import torch._inductor.ir
        import torch.utils._triton as torch_triton
        import torch.utils.flop_counter  # noqa: F401

        torch_triton.has_triton()
        torch_triton.has_triton_package()
        sys.meta_path.insert(0, _TritonFinder())


def reference_moe(x, indices, scores, w1, w2):
    """fp32 SwiGLU experts: gate rows then up rows in ``w1 [E, 2I, H]``, ``w2 [E, H, I]``."""
    t, k = indices.shape
    valid = indices >= 0
    e = indices.clamp(min=0).reshape(-1)
    xs = x.float().repeat_interleave(k, 0)
    z = torch.bmm(w1.float()[e], xs[:, :, None])[..., 0]
    gate, up = z.chunk(2, dim=-1)
    h = F.silu(gate) * up
    out = torch.bmm(w2.float()[e], h[:, :, None])[..., 0]
    weight = (scores.float() * valid).reshape(-1, 1)
    return (out * weight).view(t, k, -1).sum(1)


class FakeBuffer:
    def __init__(self, experts, max_tokens, topk, hidden, intermediate):
        self.num_max_tokens_per_rank = max_tokens
        self.x = torch.zeros(max_tokens, hidden, dtype=torch.bfloat16)
        self.x_sf = torch.zeros(max_tokens, 1)
        self.topk_idx = torch.zeros(max_tokens, topk, dtype=torch.int64)
        self.topk_weights = torch.zeros(max_tokens, topk)
        self.l1_acts = torch.zeros(max_tokens * topk, 1)
        self.buffer = torch.zeros(1)
        self.handle = types.SimpleNamespace(buffer_ptrs=[0])
        self.num_sms = None

    def destroy(self):
        pass


def _weight(t):
    return t[0] if isinstance(t, (tuple, list)) else t


def _fill_counts(counts, indices):
    if counts is not None:
        valid = indices[indices >= 0]
        counts.add_(torch.bincount(valid, minlength=counts.numel()).to(counts.dtype))


def make_deep_gemm(state):
    dg = types.ModuleType("optmoe_deepgemm")
    dg.__path__ = []

    def get_num_sms():
        return state["sms"]

    def set_num_sms(n):
        state["sms"] = n

    def get_symm_buffer_for_optmoe(
        group, experts, max_tokens, topk, hidden, inter, **_
    ):
        buffer = FakeBuffer(experts, max_tokens, topk, hidden, inter)
        buffer.num_sms = state["sms"]
        return buffer

    def optmoe(y, w1, w2, buffer, *, fast_math=False, saved_z=None, **kw):
        t = y.shape[0]
        indices = buffer.topk_idx[:t]
        y.copy_(
            reference_moe(
                buffer.x[:t], indices, buffer.topk_weights[:t], _weight(w1), _weight(w2)
            )
        )
        _fill_counts(kw.get("cumulative_local_expert_recv_stats"), indices)

    def nvfp4_optmoe(y, w1, w2, expert_scales, buffer, **kw):
        t = y.shape[0]
        indices = buffer.topk_idx[:t]
        scale = expert_scales[indices.clamp(min=0), 2] * (indices >= 0)
        y.copy_(
            (buffer.x[:t].float() * (buffer.topk_weights[:t] * scale).sum(-1, True)).to(
                y.dtype
            )
        )
        _fill_counts(kw.get("cumulative_local_expert_recv_stats"), indices)

    def fp8_forward_sm90(y, w1, w2, scales, buf, ptrs, rank, tokens, rows, topk, *rest):
        buffer = state["buffers"][-1]
        optmoe(y, w1, w2, buffer, cumulative_local_expert_recv_stats=rest[2])

    def get_symm(*a, **k):
        buffer = get_symm_buffer_for_optmoe(*a, **k)
        state.setdefault("buffers", []).append(buffer)
        return buffer

    dg.get_num_sms = get_num_sms
    dg.set_num_sms = set_num_sms
    dg.get_mk_alignment_for_contiguous_layout = lambda: 128
    dg.get_symm_buffer_for_optmoe = get_symm
    dg.bf16_optmoe = optmoe
    dg.fp8_fp4_optmoe = optmoe
    dg.nvfp4_optmoe = nvfp4_optmoe
    native = types.SimpleNamespace(
        fp8_optmoe_forward_sm90=fp8_forward_sm90,
        get_token_alignment_for_optmoe=lambda: 1,
    )
    for name in (
        "bf16_optmoe_backward_sm90",
        "bf16_optmoe_backward",
        "mxfp8_optmoe_backward",
        "fp8_optmoe_backward_sm90",
        "nvfp4_optmoe_backward",
    ):
        setattr(native, name, None)
    dg._C = native
    mega = types.ModuleType("optmoe_deepgemm.mega")

    def _interleave_weights(w):
        n = w.shape[0]
        perm = torch.arange(n).view(2, n // 16, 8).permute(1, 0, 2).reshape(-1)
        return w[perm]

    mega._interleave_weights = _interleave_weights
    dg.mega = mega
    sys.modules["optmoe_deepgemm"] = dg
    sys.modules["optmoe_deepgemm.mega"] = mega
    return dg


class FakeBound:
    """Records which forwards reported receive counts (``WorkspaceBound.record``)."""

    def __init__(self):
        self.records = []

    def record(self, key, counts):
        self.records.append(counts.clone())


def _pools(x, dy, indices, scores, w1, w2, local):
    """Pools as the native backward lays them out: each expert's rows from a 128-aligned
    start (x, score * dy, h, dz), plus fp32 dx / dscore / dW of the reference (by hand: the
    op impls run below autograd)."""
    x, dy, w1, w2 = (t.detach().float() for t in (x, dy, w1, w2))
    t, k = indices.shape
    flat = indices.reshape(-1)
    rows = torch.nonzero(flat >= 0)[:, 0]
    experts = flat[rows]
    tok = rows // k
    xs = x[tok]
    z = torch.bmm(w1[experts], xs[:, :, None])[..., 0]
    gate, up = z.chunk(2, dim=-1)
    sig = torch.sigmoid(gate)
    h = gate * sig * up
    out = torch.bmm(w2[experts], h[:, :, None])[..., 0]
    s = scores.detach().float().reshape(-1)[rows]
    dyr = dy[tok]
    dout = dyr * s[:, None]
    ds = torch.zeros(t * k).index_add(0, rows, (dyr * out).sum(-1)).view(t, k)
    dh = torch.bmm(w2[experts].transpose(1, 2), dout[:, :, None])[..., 0]
    dgate = dh * up * sig * (1 + gate * (1 - sig))
    dz = torch.cat([dgate, dh * gate * sig], -1)
    dx = torch.zeros_like(x).index_add(
        0, tok, torch.bmm(w1[experts].transpose(1, 2), dz[:, :, None])[..., 0]
    )
    dw1 = torch.zeros_like(w1).index_add(0, experts, dz[:, :, None] * xs[:, None, :])
    dw2 = torch.zeros_like(w2).index_add(0, experts, dout[:, :, None] * h[:, None, :])
    counts = torch.bincount(experts, minlength=local)
    padded = (counts + 127) // 128 * 128
    starts = torch.cumsum(padded, 0) - padded
    order = torch.argsort(experts, stable=True)
    rank_in = (
        torch.arange(len(rows)) - (torch.cumsum(counts, 0) - counts)[experts[order]]
    )
    dest = starts[experts[order]] + rank_in
    total = int(padded.sum())

    def pool(values):
        p = torch.zeros(total, values.shape[1], dtype=torch.bfloat16)
        p[dest] = values[order].detach().to(torch.bfloat16)
        return p

    pools = (
        pool(xs),
        pool(dout),
        pool(h),
        pool(dz),
    )
    return dx, ds, dw1, dw2, pools, starts.to(torch.int32), counts.to(torch.int32)


class FakeNativeBackward:
    """CPU stand-in of :class:`optmoe.persistent_backward.NativePersistentBackward`."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.arch = torch.cuda.get_device_capability()[0]
        self.blockwise = runtime.compute_dtype == "blockwise"
        self.bound = FakeBound()
        self.pools = self.starts = self.counts = None

    def precision(self):
        return "mxfp8" if self.runtime.compute_dtype == "mxfp8" else "bf16"

    def backward(self, dy, x, indices, scores, w1, w2, record_key=None, **kw):
        local = w1.shape[0]
        dx, ds, dw1, dw2, self.pools, self.starts, self.counts = _pools(
            x, dy, indices, scores, w1, w2, local
        )
        return dx.to(x.dtype), ds, dw1, dw2


class FakeNvfp4Backward:
    def __init__(self, runtime):
        self.runtime = runtime
        self.bound = FakeBound()

    def backward(self, dy, x, indices, scores, merged, key=None):
        scale = merged.scales_backward[indices.clamp(min=0), 2] * (indices >= 0)
        dx = (dy.float() * (scores * scale).sum(-1, True)).to(x.dtype)
        ds = (dy.float() * x.float()).sum(-1, True) * scale
        local = merged.scales_backward.shape[0]
        w1 = torch.zeros(local, 2 * INTER, HIDDEN)
        w2 = torch.zeros(local, HIDDEN, INTER)
        *_, self.pools, self.starts, self.counts = _pools(
            x, dy, indices, scores, w1 + 0.01, w2 + 0.01, local
        )
        return dx, ds


class _Requant:
    def __init__(self, qdata, atoms):
        self.qdata, self.scale_atoms = qdata, atoms


def _requant_one(q, a, b, transpose=False):
    e, n, k2 = q.shape
    k = 2 * k2
    if transpose:
        n, k = k, n
    delta = (torch.bmm(b.float(), a.float()).sum((1, 2)) * 0).to(torch.uint8)
    qdata = torch.zeros(e, n, k // 2, dtype=torch.uint8) + delta[:, None, None]
    atoms = torch.zeros(e, n // 128, k // 64, 512, dtype=torch.uint8)
    return _Requant(qdata, atoms)


def fake_requant_modules():
    cuda = types.SimpleNamespace(
        requant_natural=lambda q, s, pts, a, b, scaling, pts2, interleave=None: (
            _requant_one(q, a, b)
        ),
        requant_fused=lambda q, s, pts, a, b, scaling, interleave=None: (
            _requant_one(q, a, b),
            _requant_one(q, a, b, transpose=True),
        ),
    )
    return types.SimpleNamespace(requant=None, triton_requant=None, cuda_requant=cuda)


def install_stubs(path, monkey):
    """Stub the native modules and device checks for ``path``; returns the deep_gemm state."""
    for key, value in HARNESS_ENV.items():
        os.environ.setdefault(key, value)
    stub_triton()
    state = {"sms": 148}
    make_deep_gemm(state)
    arch = ARCH[path]
    sms = 132 if arch == 9 else 148
    monkey(torch.cuda, "get_device_capability", lambda device=None: (arch, 0))
    monkey(torch.cuda, "current_device", lambda: 0)
    monkey(
        torch.cuda,
        "get_device_properties",
        lambda device=None: types.SimpleNamespace(multi_processor_count=sms),
    )
    monkey(torch.cuda, "is_current_stream_capturing", lambda: False)

    import optmoe.persistent_backward as pb

    monkey(pb, "NativePersistentBackward", FakeNativeBackward)
    import optmoe.nvfp4_lora as nl

    monkey(nl, "Nvfp4PersistentBackward", FakeNvfp4Backward)
    monkey(nl, "requant_modules", fake_requant_modules)
    import optmoe.weight_cache as wc

    monkey(wc, "weight_layouts", lambda w1, w2, *names: (w1, w2))
    monkey(wc, "mxfp8_forward_weight_forms", lambda w1, w2: (w1, w2))
    import optmoe.token_quantization as tq

    def prepare_mxfp8(x, *, weights=False, interleave=False):
        if weights:
            return x
        return x, torch.zeros(x.shape[0], 1)

    def quantize_into(x, q_out, s_out, block_cols):
        q_out.copy_(x)

    monkey(tq, "prepare_persistent_mxfp8", prepare_mxfp8)
    monkey(tq, "quantize_tokens_into", quantize_into)
    import optmoe.peer_dispatch as pd
    import optmoe.sm90_prep as sp
    import optmoe.sm90_weight_forms as swf

    peer_dispatch = pd.PeerDispatch

    def local_peer_dispatch(*, capacity, top_k, hidden, group, **kwargs):
        specs = peer_dispatch.buffer_specs(1, capacity, top_k, hidden)
        workspace = pd.LocalWorkspace.create(1, "cpu", specs)[0]
        return peer_dispatch(
            capacity=capacity, top_k=top_k, hidden=hidden, workspace=workspace, **kwargs
        )

    monkey(pd, "PeerDispatch", local_peer_dispatch)
    monkey(sp, "enabled", lambda: False)
    monkey(swf, "cache_mode", lambda: "1")
    monkey(swf, "cached_forms", lambda w1, w2, names, mode: [(w1, None), (w2, None)])
    return state


def _setattr(obj, name, value):
    setattr(obj, name, value)


def init_group():
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    return dist.group.WORLD


class MoELayer(torch.nn.Module):
    def __init__(self, runtime, path, generator, state):
        super().__init__()
        self.runtime, self.path, self.state = runtime, path, state
        self.router = torch.nn.Parameter(
            torch.randn(HIDDEN, EXPERTS, generator=generator) * 0.1
        )
        lora = path.endswith("_lora")
        if path in DISPATCH:
            lora = False
        w1 = torch.randn(EXPERTS, 2 * INTER, HIDDEN, generator=generator) * 0.02
        w2 = torch.randn(EXPERTS, HIDDEN, INTER, generator=generator) * 0.02
        if not lora:
            self.w1 = torch.nn.Parameter(w1.bfloat16())
            self.w2 = torch.nn.Parameter(w2.bfloat16())
            return
        self.a1 = torch.nn.Parameter(
            torch.randn(EXPERTS, RANK, HIDDEN, generator=generator) * 0.01
        )
        self.b1 = torch.nn.Parameter(
            torch.randn(EXPERTS, 2 * INTER, RANK, generator=generator) * 0.01
        )
        self.a2 = torch.nn.Parameter(
            torch.randn(EXPERTS, RANK, INTER, generator=generator) * 0.01
        )
        self.b2 = torch.nn.Parameter(
            torch.randn(EXPERTS, HIDDEN, RANK, generator=generator) * 0.01
        )
        if path == "nvfp4_lora":
            from optmoe.nvfp4_lora import Nvfp4LoraExperts

            self.experts = Nvfp4LoraExperts(
                torch.zeros(EXPERTS, 2 * INTER, HIDDEN // 2, dtype=torch.uint8),
                torch.ones(EXPERTS, 2 * INTER, HIDDEN // 16).to(torch.float8_e4m3fn),
                torch.ones(EXPERTS),
                torch.zeros(EXPERTS, HIDDEN, INTER // 2, dtype=torch.uint8),
                torch.ones(EXPERTS, HIDDEN, INTER // 16).to(torch.float8_e4m3fn),
                torch.ones(EXPERTS),
                scaling=2.0,
            )
        else:
            from optmoe.lora import LoraExperts

            self.register_buffer("w1", w1.bfloat16())
            self.register_buffer("w2", w2.bfloat16())
            self.experts = LoraExperts(scaling=2.0)

    def forward(self, x):
        logits = x.float() @ self.router
        values, indices = logits.topk(TOPK, dim=-1)
        scores = values.softmax(-1)
        if self.path == "dispatch_bounded":

            def local_fn(recv_x, recv_idx, recv_w):
                out = reference_moe(recv_x, recv_idx, recv_w, self.w1, self.w2)
                return out.to(x.dtype)

            y = self.runtime.forward_bounded(
                x, indices, scores, EXPERTS, local_fn, id(self)
            )
        elif self.path in DISPATCH:
            recv_x, recv_idx, recv_w, handle = self.runtime.dispatch(
                x, indices, scores, EXPERTS
            )
            local = reference_moe(recv_x, recv_idx, recv_w, self.w1, self.w2)
            y = self.runtime.combine(local.to(x.dtype), handle)
        elif not self.path.endswith("_lora"):
            y = self.runtime.forward(x, indices, scores, self.w1, self.w2)
        else:
            w = {} if self.path == "nvfp4_lora" else {"w1": self.w1, "w2": self.w2}
            y = self.runtime.forward_lora(
                x,
                indices,
                scores,
                self.a1,
                self.b1,
                self.a2,
                self.b2,
                self.experts,
                self.state["cache"],
                **w,
            )
        return x + y


class Model(torch.nn.Module):
    def __init__(self, runtime, path, state, layers=2):
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.layers = torch.nn.ModuleList(
            MoELayer(runtime, path, generator, state) for _ in range(layers)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return (x.float() ** 2).mean()


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.breaks, self.recompiles = [], []

    def emit(self, record):
        message = record.getMessage()
        if "__graph_breaks" in record.name:
            self.breaks.append(message)
        elif "__recompiles" in record.name:
            self.recompiles.append(message)


@contextmanager
def capture_logs():
    torch._logging.set_logs(graph_breaks=True, recompiles=True)
    handler = _Capture()
    roots = [logging.getLogger(name) for name in ("torch", "torch._dynamo")]
    for root in roots:
        root.addHandler(handler)
    try:
        yield handler
    finally:
        for root in roots:
            root.removeHandler(handler)
        torch._logging.set_logs()


def build(path, monkey=_setattr):
    install_stubs(path, monkey)
    group = init_group()
    from optmoe.api import OptMoEDispatchRuntime, OptMoERuntime

    if path in DISPATCH:
        runtime = OptMoEDispatchRuntime(
            group=group,
            max_tokens_per_rank=MAX_TOKENS,
            top_k=TOPK,
            hidden=HIDDEN,
            compact=path != "dispatch",
            bounded=path == "dispatch_bounded",
            transfer="legacy",
        )
        return runtime, {}

    runtime = OptMoERuntime(
        group=group,
        num_experts=EXPERTS,
        max_tokens_per_rank=MAX_TOKENS,
        top_k=TOPK,
        hidden=HIDDEN,
        intermediate=INTER,
        precision=path,
    )
    state = {}
    if path == "nvfp4_lora":
        from optmoe.nvfp4_lora import RequantCache

        state["cache"] = RequantCache()
    elif path.endswith("_lora"):
        from optmoe.lora import MergedWeightCache

        state["cache"] = MergedWeightCache()
    return runtime, state


def run_steps(model, state, step_fn, steps=STEP_TOKENS, seed=1, after_step=None):
    generator = torch.Generator().manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.1)
    if "cache" in state:
        from optmoe.lora import attach_post_step_hook

        hook = attach_post_step_hook(optimizer, state["cache"])
    losses, grads = [], []
    for tokens in steps:
        x = torch.randn(tokens, HIDDEN, generator=generator).bfloat16().requires_grad_()
        loss = step_fn(x)
        loss.backward()
        losses.append(loss.detach())
        grads.append([p.grad.detach().clone() for p in params] + [x.grad.clone()])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if after_step is not None:
            after_step()
    if "cache" in state:
        hook.remove()
    return losses, grads


def count_compiles():
    calls = []

    def backend(gm, example_inputs):
        from functorch.compile import make_boxed_func
        from torch._dynamo.backends.common import aot_autograd

        calls.append(gm)
        boxed = lambda g, _: make_boxed_func(g)
        return aot_autograd(fw_compiler=boxed, bw_compiler=boxed)(gm, example_inputs)

    return backend, calls


def fresh_model(runtime, path, state, layers, warm):
    """``warm``: one eager forward first, as a trainer's prepare or eager step would run, so
    the layers' registrations exist before tracing (LoRA layers first seen inside a compiled
    region register through a one-time graph break). ``"register"`` registers them with
    ``register_layer`` instead."""
    model = Model(runtime, path, state, layers)
    if warm == "register":
        from optmoe.lora import register_layer

        for layer in model.layers:
            register_layer(layer.experts, state["cache"])
    elif warm:
        with torch.no_grad():
            model(torch.randn(8, HIDDEN).bfloat16())
    return model


def analyze(path, *, layers=2, steps=STEP_TOKENS, fullgraph=True, warm=False):
    """Explain + fullgraph compile of forward + backward for ``path``; a JSON-able dict."""
    import torch._dynamo

    runtime, state = build(path)
    result = {"path": path}

    torch._dynamo.reset()
    model = fresh_model(runtime, path, state, layers, warm)
    x = torch.randn(steps[0], HIDDEN).bfloat16().requires_grad_()
    explanation = torch._dynamo.explain(model)(x)
    result["graph_count"] = explanation.graph_count
    result["graph_break_count"] = explanation.graph_break_count
    result["break_reasons"] = [
        {
            "reason": r.reason.splitlines()[0][:300],
            "frames": [f"{f.filename}:{f.lineno}" for f in r.user_stack[-4:]],
        }
        for r in explanation.break_reasons
    ]

    torch._dynamo.reset()
    eager_model = Model(runtime, path, state, layers)
    eager_losses, eager_grads = run_steps(eager_model, state, eager_model)

    torch._dynamo.reset()
    model = fresh_model(runtime, path, state, layers, warm)
    backend, calls = count_compiles()
    compiled = torch.compile(model, backend=backend, fullgraph=fullgraph)
    with capture_logs() as logs:
        try:
            per_step = []
            losses, grads = run_steps(
                model,
                state,
                compiled,
                steps,
                after_step=lambda: per_step.append(len(logs.breaks)),
            )
            result["breaks_after_step"] = per_step
            result["compile_error"] = None
        except Exception as error:  # noqa: BLE001
            result["compile_error"] = f"{type(error).__name__}: {str(error)[:2000]}"
            losses = grads = None
    result["graphs_compiled"] = len(calls)
    result["recompiles"] = len(logs.recompiles)
    result["recompile_reasons"] = [m[:500] for m in logs.recompiles]
    result["logged_breaks"] = [m[:500] for m in logs.breaks]
    if losses is not None:
        result["max_loss_diff"] = max(
            float((a - b).abs()) for a, b in zip(losses, eager_losses, strict=True)
        )
        result["max_grad_diff"] = max(
            float((a.float() - b.float()).abs().max())
            for ga, gb in zip(grads, eager_grads, strict=True)
            for a, b in zip(ga, gb, strict=True)
        )
    bound = _bound(runtime)
    result["records"] = None if bound is None else len(bound.records)
    runtime.destroy()
    return result


def _bound(runtime):
    moe = getattr(runtime, "_moe", None)
    if moe is None:
        return None
    return getattr(moe._peer_backward, "bound", None) or getattr(
        getattr(moe, "_nvfp4_backward", None), "bound", None
    )


def records_after_eval(path, *, layers=2):
    """Receive-count records pushed by each compiled forward of: train, eval (``no_grad``),
    train. Every training forward must record once per layer, every eval forward never."""
    import torch._dynamo

    runtime, state = build(path)
    torch._dynamo.reset()
    model = Model(runtime, path, state, layers)
    model(torch.randn(8, HIDDEN).bfloat16())
    compiled = torch.compile(model, backend=count_compiles()[0])
    pushed = []
    for grad in (True, False, True):
        bound = _bound(runtime)
        before = 0 if bound is None else len(bound.records)
        x = torch.randn(16, HIDDEN).bfloat16().requires_grad_()
        with torch.set_grad_enabled(grad):
            loss = compiled(x)
        bound = _bound(runtime)
        pushed.append(len(bound.records) - before)
        if grad:
            loss.backward()
    runtime.destroy()
    return {"path": path, "pushed": pushed, "expected": [layers, 0, layers]}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("path", choices=PATHS)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--no-fullgraph", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--records", action="store_true")
    parser.add_argument("--warm", action="store_true")
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args(argv)
    if args.records:
        result = records_after_eval(args.path, layers=args.layers)
    else:
        result = analyze(
            args.path,
            layers=args.layers,
            fullgraph=not args.no_fullgraph,
            warm="register" if args.register else args.warm,
        )
    if args.json:
        print("HARNESS_RESULT " + json.dumps(result))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()

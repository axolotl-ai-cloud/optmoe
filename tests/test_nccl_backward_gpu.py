# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""The BF16 backward the runtime falls back to when the persistent backward does not cover the
shape (``backward_backend="nccl"``) matches the persistent backward on Hopper and Blackwell."""

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.gpu_blackwell]

HIDDEN, INNER, EXPERTS, TOPK, TOKENS = 1024, 512, 8, 4, 256


@pytest.fixture
def group(tmp_path):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (
        9,
        10,
    ):
        pytest.skip("the fused runtime needs SM90 or SM100")
    pytest.importorskip("optmoe_deepgemm")
    import torch.distributed as dist

    dist.init_process_group(
        "nccl",
        store=dist.FileStore(str(tmp_path / "store"), 1),
        rank=0,
        world_size=1,
        device_id=torch.device("cuda", torch.cuda.current_device()),
    )
    yield dist.group.WORLD
    dist.destroy_process_group()


def _runtime(group, backend, inner=INNER):
    from optmoe._native import deep_gemm
    from optmoe.runtime import PersistentMoE

    deep_gemm.set_num_sms(torch.cuda.get_device_properties(0).multi_processor_count)
    return PersistentMoE(
        group,
        EXPERTS,
        TOKENS,
        TOPK,
        HIDDEN,
        inner,
        compute_dtype="bf16",
        fuse_activation=True,
        backward_backend=backend,
    )


def _inputs(inner=INNER):
    gen = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(TOKENS, HIDDEN, device="cuda", generator=gen).bfloat16()
    ids = torch.rand(TOKENS, EXPERTS, device="cuda", generator=gen).argsort(-1)[
        :, :TOPK
    ]
    scores = torch.rand(TOKENS, TOPK, device="cuda", generator=gen).softmax(-1)
    w1 = (
        torch.randn(EXPERTS, 2 * inner, HIDDEN, device="cuda", generator=gen)
        / HIDDEN**0.5
    ).bfloat16()
    w2 = (
        torch.randn(EXPERTS, HIDDEN, inner, device="cuda", generator=gen) / inner**0.5
    ).bfloat16()
    dy = torch.randn(TOKENS, HIDDEN, device="cuda", generator=gen).bfloat16()
    return x, ids.contiguous(), scores, w1, w2, dy


def _step(runtime, inputs):
    x, ids, scores, w1, w2, dy = inputs
    x, scores = x.clone().requires_grad_(), scores.clone().requires_grad_()
    w1, w2 = w1.clone().requires_grad_(), w2.clone().requires_grad_()
    y = runtime(x, ids, scores, w1, w2)
    y.backward(dy)
    torch.cuda.synchronize()
    return [t.float() for t in (y.detach(), x.grad, scores.grad, w1.grad, w2.grad)]


def _relative(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def test_nccl_backward_matches_the_persistent_backward(group):
    inputs = _inputs()
    native = _runtime(group, "native")
    reference = _step(native, inputs)
    native.close()
    fallback = _runtime(group, "nccl")
    result = _step(fallback, inputs)
    fallback.close()
    for name, got, want in zip(
        ("y", "dx", "dscore", "dw1", "dw2"), result, reference, strict=True
    ):
        assert torch.isfinite(got).all(), name
        assert _relative(got, want) < 2e-2, (name, _relative(got, want))


def test_nccl_backward_runs_a_shape_the_persistent_backward_does_not_cover(group):
    # an intermediate size that is not a multiple of 256 is what selects this backward
    inner = 128
    runtime = _runtime(group, "nccl", inner=inner)
    result = _step(runtime, _inputs(inner))
    runtime.close()
    for name, tensor in zip(("y", "dx", "dscore", "dw1", "dw2"), result, strict=True):
        assert torch.isfinite(tensor).all(), name
        assert tensor.abs().max() > 0, name

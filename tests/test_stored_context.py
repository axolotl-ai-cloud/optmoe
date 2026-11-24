# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Stored forward context and the bounded workspace on CPU, against a stand-in ``deep_gemm``."""

import sys
from types import SimpleNamespace

import pytest
import torch

from optmoe import runtime as runtime_module
from optmoe import workspace_bound
from optmoe.workspace_bound import WorkspaceBound


def _pinned_memory():
    try:
        torch.empty((), pin_memory=True)
    except RuntimeError:
        return False
    return True


# the bound's host read is pinned, which needs an accelerator backend even with faked kernels
needs_pinned_memory = pytest.mark.skipif(
    not _pinned_memory(), reason="no pinned memory allocator"
)

EXPERTS, HIDDEN, INTERMEDIATE, TOPK, TOKENS = 2, 256, 256, 2, 64
COUNTS = (130, 5)


class _Event:
    def record(self, *args):
        pass

    def synchronize(self):
        pass

    def query(self):
        return True


def _setup(monkeypatch, policy, graph_task=-1):
    monkeypatch.setenv("OPTMOE_STORED_CONTEXT", policy)
    monkeypatch.delenv("OPTMOE_STORED_CONTEXT_ROWS", raising=False)
    monkeypatch.delenv("OPTMOE_SM100_FORWARD_PHASE_TRACE", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "optmoe_deepgemm",
        SimpleNamespace(get_num_sms=lambda: 4, bf16_optmoe="bf16_optmoe"),
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 0))
    monkeypatch.setattr(torch.cuda, "Event", _Event)
    monkeypatch.setattr(torch._C, "_current_graph_task_id", lambda: graph_task)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 1)
    monkeypatch.setattr(
        "optmoe.weight_cache.weight_layouts", lambda w1, w2, *names: (w1, w2)
    )
    calls = []

    def launch(runtime, direction, kernel, y, tw1, tw2, buffer, **kwargs):
        counts = kwargs.get("cumulative_local_expert_recv_stats")
        if counts is not None:
            counts.copy_(torch.tensor(COUNTS, dtype=torch.int32))
        calls.append(kwargs)

    monkeypatch.setattr(runtime_module, "launch", launch)
    worst = TOKENS * TOPK + EXPERTS * 128
    buffer = SimpleNamespace(
        num_sms=4,
        num_max_tokens_per_rank=TOKENS,
        x=torch.empty(TOKENS, HIDDEN, dtype=torch.bfloat16),
        topk_idx=torch.empty(TOKENS, TOPK, dtype=torch.int64),
        topk_weights=torch.empty(TOKENS, TOPK),
    )
    rt = SimpleNamespace(
        num_sms=4,
        buffer=buffer,
        compute_dtype="bf16",
        retain_activations=False,
        group=None,
        experts=EXPERTS,
        max_tokens=TOKENS,
        topk=TOPK,
        hidden=HIDDEN,
        intermediate=INTERMEDIATE,
        _record_forward=True,
        _stored=None,
        _peer_backward=SimpleNamespace(
            stored_context_usable=lambda: True,
            precision=lambda: "bf16",
            bound=WorkspaceBound(TOKENS, TOPK, EXPERTS, worst),
        ),
    )
    return rt, calls


def _inputs():
    x = torch.randn(16, HIDDEN, dtype=torch.bfloat16)
    indices = torch.zeros(16, TOPK, dtype=torch.int64)
    scores = torch.rand(16, TOPK)
    w1 = torch.randn(EXPERTS, 2 * INTERMEDIATE, HIDDEN, dtype=torch.bfloat16)
    w2 = torch.randn(EXPERTS, HIDDEN, INTERMEDIATE, dtype=torch.bfloat16)
    return x, indices, scores, w1, w2


@needs_pinned_memory
def test_always_storing_forward_records_the_bounded_workspace(monkeypatch):
    rt, calls = _setup(monkeypatch, "always")
    x, indices, scores, w1, w2 = _inputs()
    runtime_module._forward_impl(x, indices, scores, w1, w2, rt)
    assert calls[0]["saved_z"] is rt._stored.buffer
    assert "cumulative_local_expert_recv_stats" in calls[0]
    assert rt._stored.key is not None
    bound = rt._peer_backward.bound
    # the backward sizes its pools to the routed rows, not the worst case
    assert bound.required_rows(w1) == 256 + 128
    assert bound.unrecorded == 0


def test_always_storing_forward_without_backward_leaves_no_record(monkeypatch):
    rt, calls = _setup(monkeypatch, "always")
    rt._record_forward = False
    x, indices, scores, w1, w2 = _inputs()
    runtime_module._forward_impl(x, indices, scores, w1, w2, rt)
    assert "cumulative_local_expert_recv_stats" not in calls[0]
    assert not rt._peer_backward.bound.records


@needs_pinned_memory
@pytest.mark.parametrize("policy", ["auto", "recompute"])
def test_checkpoint_recompute_storing_forward_keeps_the_original_record(
    monkeypatch, policy
):
    rt, calls = _setup(monkeypatch, policy)
    x, indices, scores, w1, w2 = _inputs()
    runtime_module._forward_impl(x, indices, scores, w1, w2, rt)
    assert "saved_z" not in calls[0]
    assert "cumulative_local_expert_recv_stats" in calls[0]
    monkeypatch.setattr(torch._C, "_current_graph_task_id", lambda: 0)
    monkeypatch.setattr(
        "optmoe.stored_context.StoredContext._fits", lambda self, n: (True, 1 << 40)
    )
    monkeypatch.setattr("optmoe.stored_context.shape_rule", lambda h, i: True)
    runtime_module._forward_impl(x, indices, scores, w1, w2, rt)
    assert calls[1]["saved_z"] is rt._stored.buffer
    assert "cumulative_local_expert_recv_stats" not in calls[1]
    bound = rt._peer_backward.bound
    assert len(bound.records[workspace_bound.layer_key(w1)]) == 1
    assert bound.required_rows(w1) == 256 + 128


def _stored_context(monkeypatch, rows):
    from optmoe.stored_context import StoredContext

    if rows is None:
        monkeypatch.delenv("OPTMOE_STORED_CONTEXT_ROWS", raising=False)
    else:
        monkeypatch.setenv("OPTMOE_STORED_CONTEXT_ROWS", rows)
    rt = SimpleNamespace(
        buffer=SimpleNamespace(num_max_tokens_per_rank=TOKENS),
        experts=EXPERTS,
        intermediate=INTERMEDIATE,
        topk=TOPK,
        max_tokens=TOKENS,
    )
    return StoredContext(rt, 1)


@pytest.mark.parametrize("rows", [None, ""])
def test_stored_context_rows_unset_or_empty_keeps_the_growing_default(
    monkeypatch, rows
):
    stored = _stored_context(monkeypatch, rows)
    assert stored.capacity == 256 + EXPERTS * 128
    assert not stored.fixed_capacity


def test_stored_context_rows_override_is_fixed(monkeypatch):
    stored = _stored_context(monkeypatch, "4096")
    assert stored.capacity == 4096
    assert stored.fixed_capacity


@pytest.mark.parametrize("rows", ["-1", "1k"])
def test_stored_context_rows_rejects_invalid_values(monkeypatch, rows):
    with pytest.raises(ValueError, match="OPTMOE_STORED_CONTEXT_ROWS"):
        _stored_context(monkeypatch, rows)

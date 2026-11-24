# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

import sys
import types

import torch

from optmoe.peer_dispatch import LocalWorkspace, PeerDispatch


class _FakeCuda(torch.Tensor):
    @property
    def is_cuda(self):
        return True


def test_legacy_compact_push_passes_matching_row_counts(monkeypatch):
    calls = []

    def recorder(dx, ds, row_assignments, destinations, score_destinations):
        calls.append((dx.shape[0], ds.shape[0], row_assignments.shape[0]))

    monkeypatch.setitem(
        sys.modules,
        "optmoe.peer_packing",
        types.SimpleNamespace(return_peer_rows=recorder),
    )
    world, capacity, top_k, hidden = 2, 4, 2, 8
    workspace = LocalWorkspace.create(
        world, "cpu", PeerDispatch.buffer_specs(world, capacity, top_k, hidden)
    )
    runtime = PeerDispatch(
        capacity=capacity,
        top_k=top_k,
        hidden=hidden,
        workspace=workspace[0],
        transfer="legacy",
        compact=True,
    )
    payload = torch.randn(2, hidden).as_subclass(_FakeCuda)
    runtime.push(payload, torch.tensor([1, 5]), "y")
    assert calls == [(2, 2, 2)]

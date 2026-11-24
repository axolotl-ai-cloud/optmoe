# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Cache keys that must not alias: the merged-weight cache across layer objects, and the
prepared set across ``OPTMOE_*`` kernel selectors. CPU only."""

import itertools
from types import SimpleNamespace

import torch
from optmoe import prepare
from optmoe.lora import MergedWeightCache

_KEYS = itertools.count()


class _Experts:
    def __init__(self):
        self.key = next(_KEYS)

    def merged(self, *adapters, version):
        return SimpleNamespace(
            owner=self.key,
            version=version,
            has_forward=True,
            has_backward=True,
            nbytes=lambda: 0,
        )


def test_merged_cache_does_not_serve_a_dead_layer_to_a_new_object():
    cache = MergedWeightCache(mode="all")
    adapters = [torch.zeros(1)]
    stale = cache.get(_Experts(), adapters)
    # an object id can be reused once the first layer is collected; its key cannot
    for _ in range(64):
        experts = _Experts()
        entry = cache.get(experts, adapters)
        assert entry.owner == experts.key
        assert entry is not stale
        assert experts.key in cache.entries
        assert id(experts) not in cache.entries
    assert cache.hits == 0


def test_merged_cache_hits_for_the_same_layer_until_invalidated():
    cache = MergedWeightCache(mode="all")
    experts, adapters = _Experts(), [torch.zeros(1)]
    entry = cache.get(experts, adapters)
    assert cache.get(experts, adapters) is entry
    assert cache.hits == 1
    cache.invalidate()
    assert cache.get(experts, adapters) is not entry


def test_prepared_key_follows_the_kernel_selectors(monkeypatch):
    monkeypatch.delenv("OPTMOE_LORA_WGRAD", raising=False)
    before = prepare._selectors()
    assert prepare._selectors() == before
    monkeypatch.setenv("OPTMOE_LORA_WGRAD", "grouped")
    changed = prepare._selectors()
    assert changed != before
    assert ("OPTMOE_LORA_WGRAD", "grouped") in changed
    monkeypatch.setenv("UNRELATED", "1")
    assert prepare._selectors() == changed

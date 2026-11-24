# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""``0`` / ``1`` environment switches reject anything else. CPU only."""

import pytest
import torch
from optmoe import weight_cache
from optmoe.env import flag


@pytest.mark.parametrize("default", [False, True])
def test_flag_reads_zero_one_and_the_default(monkeypatch, default):
    monkeypatch.delenv("OPTMOE_TEST_FLAG", raising=False)
    assert flag("OPTMOE_TEST_FLAG", default) is default
    monkeypatch.setenv("OPTMOE_TEST_FLAG", "0")
    assert flag("OPTMOE_TEST_FLAG", default) is False
    monkeypatch.setenv("OPTMOE_TEST_FLAG", "1")
    assert flag("OPTMOE_TEST_FLAG", default) is True


@pytest.mark.parametrize("value", ["", "true", "false", "2", "off", " 1"])
def test_flag_rejects_other_values(monkeypatch, value):
    monkeypatch.setenv("OPTMOE_TEST_FLAG", value)
    with pytest.raises(ValueError, match="OPTMOE_TEST_FLAG must be 0 or 1"):
        flag("OPTMOE_TEST_FLAG", True)


def test_weight_cache_build_rejects_a_misspelled_switch(monkeypatch):
    w = torch.zeros(1, 2, 2)
    monkeypatch.setenv("OPTMOE_WEIGHT_CACHE_COMPILED", "false")
    with pytest.raises(ValueError, match="OPTMOE_WEIGHT_CACHE_COMPILED"):
        weight_cache._build("w1", w, w)
    monkeypatch.setenv("OPTMOE_WEIGHT_CACHE_COMPILED", "1")
    monkeypatch.setenv("OPTMOE_WEIGHT_CACHE_FUSED", "yes")
    with pytest.raises(ValueError, match="OPTMOE_WEIGHT_CACHE_FUSED"):
        weight_cache._build_missing(("w1", "w1_t"), w, w)

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""``OPTMOE_SM100_MXFP8_FORMS_CONFIG`` parsing (CPU; a stand-in ``triton`` when it is not installed)."""

import importlib
import sys
from types import ModuleType

import pytest


@pytest.fixture
def mx_weight_forms(monkeypatch):
    try:
        import triton  # noqa: F401
    except ImportError:
        language = ModuleType("triton.language")
        language.constexpr = object
        stub = ModuleType("triton")
        stub.jit = lambda fn: fn
        stub.language = language
        monkeypatch.setitem(sys.modules, "triton", stub)
        monkeypatch.setitem(sys.modules, "triton.language", language)
        monkeypatch.delitem(sys.modules, "optmoe.mx_weight_forms", raising=False)
        module = importlib.import_module("optmoe.mx_weight_forms")
        # the stand-in import must not outlive the test
        monkeypatch.setitem(sys.modules, "optmoe.mx_weight_forms", module)
        return module
    return importlib.import_module("optmoe.mx_weight_forms")


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, (2, 32, True)),
        ("4,32,0", (4, 32, False)),
        ("8/64/1", (8, 64, True)),
        ("4,128,0", (4, 128, False)),
    ],
)
def test_forms_config_accepts_valid_values(
    monkeypatch, mx_weight_forms, value, expected
):
    if value is None:
        monkeypatch.delenv("OPTMOE_SM100_MXFP8_FORMS_CONFIG", raising=False)
    else:
        monkeypatch.setenv("OPTMOE_SM100_MXFP8_FORMS_CONFIG", value)
    assert mx_weight_forms._launch_config() == expected


# pass_rows 256 runs zero passes per tile and leaves the forms uninitialized
@pytest.mark.parametrize(
    "value", ["2,256,1", "2,16,1", "2,48,1", "3,32,1", "2,32,2", "2,32"]
)
def test_forms_config_rejects_invalid_values(monkeypatch, mx_weight_forms, value):
    monkeypatch.setenv("OPTMOE_SM100_MXFP8_FORMS_CONFIG", value)
    with pytest.raises(ValueError, match="OPTMOE_SM100_MXFP8_FORMS_CONFIG"):
        mx_weight_forms._launch_config()

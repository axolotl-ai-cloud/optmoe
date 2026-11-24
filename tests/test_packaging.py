# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""The kernel extension's install naming and the message when it is missing. CPU only."""

import sys

import pytest
import torch
from optmoe import _native, install_kernels


def test_wheel_url_names_python_torch_and_cuda(monkeypatch):
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(torch, "__version__", "2.14.1+cu130")
    monkeypatch.setattr(install_kernels.platform, "machine", lambda: "x86_64")
    python = f"cp{sys.version_info.major}{sys.version_info.minor}"
    name = install_kernels.wheel_name("0.1.0")
    assert name == (
        f"optmoe_deepgemm-0.1.0+cu13.torch2.14-{python}-{python}-linux_x86_64.whl"
    )
    assert install_kernels.wheel_url("0.1.0", "org/repo") == (
        "https://github.com/org/repo/releases/download/v0.1.0/"
        + name.replace("+", "%2B")
    )


def test_cpu_torch_is_rejected(monkeypatch):
    monkeypatch.setattr(torch.version, "cuda", None)
    with pytest.raises(RuntimeError, match="CUDA build of torch"):
        install_kernels.build_tag()


def test_missing_extension_names_the_install_command(monkeypatch):
    monkeypatch.setitem(sys.modules, "optmoe_deepgemm", None)
    with pytest.raises(ModuleNotFoundError, match="optmoe.install_kernels"):
        _native.deep_gemm  # noqa: B018


def test_extension_is_resolved_on_every_access(monkeypatch):
    first, second = object(), object()
    monkeypatch.setitem(sys.modules, "optmoe_deepgemm", first)
    assert _native.deep_gemm is first
    monkeypatch.setitem(sys.modules, "optmoe_deepgemm", second)
    assert _native.deep_gemm is second

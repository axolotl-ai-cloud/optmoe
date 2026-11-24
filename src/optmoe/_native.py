# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""The compiled kernel extension, ``optmoe_deepgemm``: OPTMoE's DeepGEMM fork, installed
separately from the pure-Python package (``python -m optmoe.install_kernels``)."""

import importlib

MISSING = (
    "optmoe needs its compiled kernel extension, optmoe_deepgemm, which is not installed. "
    "Install it with `python -m optmoe.install_kernels` (a prebuilt wheel for this Python, "
    "torch and CUDA version, or a source build)."
)


def __getattr__(name):
    # resolved on every access, so the extension loads lazily and tests can swap it
    if name != "deep_gemm":
        raise AttributeError(name)
    try:
        return importlib.import_module("optmoe_deepgemm")
    except ModuleNotFoundError as error:
        if error.name != "optmoe_deepgemm":
            raise
        raise ModuleNotFoundError(MISSING, name="optmoe_deepgemm") from None

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Host-side argument validation of the native extensions, on meta tensors (no GPU)."""

import pytest
import torch


def _owned_operands(rows, hidden, inner, experts):
    def t(*shape, dtype=torch.bfloat16):
        return torch.empty(*shape, device="meta", dtype=dtype)

    return (
        t(rows, hidden),
        t(rows, hidden),
        t(experts, 2 * inner, hidden),
        t(experts, hidden, inner),
        t(rows, dtype=torch.float32),
        t(experts, dtype=torch.int32),
        t(experts, dtype=torch.int32),
    )


@pytest.mark.parametrize(
    "rows, hidden, inner, experts",
    [
        (16384 * 8 * 8, 5120, 1536, 16),  # rows * hidden
        ((1 << 18) + 1, 2048, 8192, 8),  # rows * 2 * inner
        (128, 5120, 16384, 32),  # experts * 2 * inner * hidden
    ],
)
def test_owned_backward_rejects_32bit_index_overflow(rows, hidden, inner, experts):
    from optmoe.owned_backward import _validate

    with pytest.raises(ValueError, match="32-bit"):
        _validate(*_owned_operands(rows, hidden, inner, experts))


@pytest.mark.parametrize(
    "rows, hidden, inner, experts",
    [
        (4096, 5120, 1536, 16),
        (16384 * 8 * 8, 2048, 768, 16),  # 30B, every routed row on one rank
        (16384 * 8 * 8, 4096, 1536, 16),  # 235B: rows * hidden == 2**32
        (131072, 5120, 2560, 32),  # Beam, balanced EP8 routing
    ],
)
def test_owned_backward_in_range_shape_reaches_device_check(
    rows, hidden, inner, experts
):
    from optmoe.owned_backward import _validate

    with pytest.raises(ValueError, match="Invalid SM100 owned-backward"):
        _validate(*_owned_operands(rows, hidden, inner, experts))


def _requant_operands(E=4, N=256, K=512, r=16, a_device="meta"):
    q = torch.empty(E, N, K // 2, device="meta", dtype=torch.uint8)
    s = torch.empty(E, N, K // 16, device="meta", dtype=torch.uint8)
    a = torch.empty(E, r, K, device=a_device, dtype=torch.bfloat16)
    b = torch.empty(E, N, r, device="meta", dtype=torch.bfloat16)
    return q, s, a, b


@pytest.mark.parametrize("e0, e1", [(0, 5), (-1, 2), (3, 2)])
def test_cuda_requant_rejects_expert_range_outside_the_base(e0, e1):
    from optmoe.nvfp4_requant.cuda_nvfp4_requant import (
        requant_fused,
        requant_natural,
    )

    q, s, a, b = _requant_operands()
    with pytest.raises(ValueError, match="expert range"):
        requant_natural(q, s, None, a, b, 1.0, None, e0, e1)
    pts = torch.ones(4, device="meta")
    with pytest.raises(ValueError, match="expert range"):
        requant_fused(q, s, pts, a, b, 1.0, e0, e1)


@pytest.mark.parametrize("e0, e1", [(0, 5), (-1, 2), (3, 2)])
def test_triton_requant_rejects_expert_range_outside_the_base(e0, e1):
    from optmoe.nvfp4_requant.triton_nvfp4_requant import requant_lora_packed

    E, N, K, r = 4, 256, 512, 16
    q, s, _, _ = _requant_operands(E, N, K, r)
    lora_a = torch.empty(E * r, K, device="meta", dtype=torch.bfloat16)
    lora_b = torch.empty(N, r * E, device="meta", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="expert range"):
        requant_lora_packed(q, s, None, lora_a, lora_b, 1.0, None, e0, e1)


def test_cuda_requant_rejects_operands_on_another_device():
    from optmoe.nvfp4_requant.cuda_nvfp4_requant import requant_natural

    q, s, a, b = _requant_operands(a_device="cpu")
    with pytest.raises(ValueError, match="one device"):
        requant_natural(q, s, None, a, b, 1.0, None)

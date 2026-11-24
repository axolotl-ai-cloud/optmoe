# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def test_scatter_add_past_32_bit_element_offsets():
    from optmoe.combine import peer_scatter_add_publish

    rows, hidden, tokens = 270_000, 8192, 64
    if torch.cuda.mem_get_info()[0] < 8 * 2**30:
        pytest.skip("needs 8 GiB of free GPU memory")
    assert rows * hidden > 2**31
    source = torch.zeros(rows, hidden, device="cuda", dtype=torch.bfloat16)
    source[-1024:] = 1.0
    indices = torch.full((rows,), -1, device="cuda", dtype=torch.int32)
    indices[-1024:] = torch.arange(1024, device="cuda", dtype=torch.int32) % tokens
    destination = torch.zeros(tokens, hidden, device="cuda")
    queue = torch.zeros(1, 4, device="cuda", dtype=torch.int32)
    peer_scatter_add_publish(source, indices, destination, queue, 0)
    assert queue[0, 3].item() == 1
    assert torch.equal(destination, torch.full_like(destination, 1024 // tokens))

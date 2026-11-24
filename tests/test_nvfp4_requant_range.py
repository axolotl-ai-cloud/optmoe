# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""The packed NVFP4 grid at a fixed per-tensor scale tops out at the amax that scale came from."""

import pytest
import torch

from optmoe.nvfp4_requant import nvfp4_requant as rq


def _roundtrip(w, pts):
    rows = pts.expand(w.shape[0])
    packed, scale = rq.quantize_nvfp4_packed_ref(w, rows)
    return rq.dequantize_nvfp4_packed_ref(packed, scale, rows, torch.float32)


def test_fixed_per_tensor_scale_clamps_at_base_amax():
    torch.manual_seed(0)
    base = torch.randn(8, 64) * 0.02
    amax = base.abs().amax()
    pts = rq.per_tensor_scale_from_amax(amax)
    assert _roundtrip(base, pts).abs().amax() <= amax * (1 + 1e-6)

    grown = base * 4
    out = _roundtrip(grown, pts)
    assert out.abs().amax() <= amax * (1 + 1e-6)
    assert ((out - grown).norm() / grown.norm()) > 0.3

    inside = base * 0.5
    assert ((_roundtrip(inside, pts) - inside).norm() / inside.norm()) < 0.15


def _packed_base(e=2, n=8, k=64, std=0.02):
    w = torch.randn(e, n, k) * std
    pts = rq.per_tensor_scale_from_amax(w.abs().amax(dim=(1, 2)))
    rows = rq.expand_pts_rows(pts, n)
    packed, scale = rq.quantize_nvfp4_packed_ref(w.reshape(e * n, k), rows)
    return w, packed.view(e, n, k // 2), scale.view(e, n, k // 16), pts


def _dequant(packed, scale, pts):
    e, n, k2 = packed.shape
    return rq.dequantize_nvfp4_packed_ref(
        packed.reshape(e * n, k2),
        scale.reshape(e * n, -1),
        rq.expand_pts_rows(pts, n),
        torch.float32,
    ).view(e, n, 2 * k2)


def test_widen_per_tensor_scale_keeps_the_base_and_raises_the_clamp():
    torch.manual_seed(1)
    w, packed, scale, pts = _packed_base()
    base = _dequant(packed, scale, pts)
    wide_q, wide_s, wide_pts, reencoded = rq.widen_per_tensor_scale(
        packed, scale, pts, 2
    )
    assert reencoded == 0
    assert torch.equal(wide_q, packed)
    assert torch.equal(wide_pts, pts * 4)
    assert torch.equal(_dequant(wide_q, wide_s, wide_pts), base)

    e, n, k = w.shape
    grown = (base * 3).reshape(e * n, k)
    for p, clipped in ((pts, True), (wide_pts, False)):
        rows = rq.expand_pts_rows(p, n)
        q, s = rq.quantize_nvfp4_packed_ref(grown, rows)
        out = rq.dequantize_nvfp4_packed_ref(q, s, rows, torch.float32)
        err = (out - grown).norm() / grown.norm()
        assert (err > 0.3) == clipped


def test_widen_per_tensor_scale_reencodes_blocks_below_the_scale_floor():
    torch.manual_seed(2)
    w, _, _, pts = _packed_base(e=1)
    w[0, 0, :16] *= 2.0**-15
    e, n, k = w.shape
    rows = rq.expand_pts_rows(pts, n)
    packed, scale = rq.quantize_nvfp4_packed_ref(
        w.reshape(e * n, k), rows, modelopt_compat=True
    )
    packed, scale = packed.view(e, n, k // 2), scale.view(e, n, k // 16)
    base = _dequant(packed, scale, pts)
    wide_q, wide_s, wide_pts, reencoded = rq.widen_per_tensor_scale(
        packed, scale, pts, 3
    )
    assert reencoded >= 1
    out = _dequant(wide_q, wide_s, wide_pts)
    changed = (out != base).reshape(e, n, -1, 16).any(-1)
    assert int(changed.sum()) <= reencoded
    step = wide_pts.max() * rq.MODELOPT_SCALE_FLOOR
    assert (out - base).abs().max() <= step


def test_widen_per_tensor_scale_zero_is_identity():
    _, packed, scale, pts = _packed_base()
    q, s, p, reencoded = rq.widen_per_tensor_scale(packed, scale, pts, 0)
    assert q is packed and s is scale and reencoded == 0
    assert torch.equal(p, pts)


def test_experts_headroom_scales_the_kernel_tables(monkeypatch):
    nvfp4_lora = pytest.importorskip("optmoe.nvfp4_lora")
    monkeypatch.setattr(
        nvfp4_lora, "interleave_perm", lambda n2, device: torch.arange(n2)
    )
    torch.manual_seed(3)
    _, q1, s1, pts1 = _packed_base(e=2, n=16, k=32)
    _, q2, s2, pts2 = _packed_base(e=2, n=32, k=16)

    def build(**kwargs):
        return nvfp4_lora.Nvfp4LoraExperts(
            q1, s1, pts1, q2, s2, pts2, scaling=2.0, backend="cuda_fused", **kwargs
        )

    plain, wide = build(), build(headroom_log2=1)
    assert plain.headroom_log2 == 0 and torch.equal(plain.pts1, pts1)
    assert wide.reencoded_blocks == 0
    assert torch.equal(wide.pts1, pts1 * 2) and torch.equal(wide.pts2, pts2 * 2)
    assert torch.equal(wide.scales_forward[:, 2], plain.scales_forward[:, 2] * 2)
    assert torch.equal(wide.scales_backward[:, 3], plain.scales_backward[:, 3] * 2)
    assert torch.equal(_dequant(wide.q1, wide.s1, wide.pts1), _dequant(q1, s1, pts1))
    monkeypatch.setenv("OPTMOE_NVFP4_LORA_HEADROOM_LOG2", "2")
    assert build().headroom_log2 == 2
    with pytest.raises(ValueError):
        build(headroom_log2=5)


@pytest.mark.parametrize("scale_mode", ["fresh", "reuse"])
def test_requant_on_widened_base_matches_until_the_base_clamps(scale_mode):
    torch.manual_seed(4)
    e, n, k, r = 2, 8, 64, 4
    _, packed, scale, pts = _packed_base(e=e, n=n, k=k)
    wide_q, wide_s, wide_pts, _ = rq.widen_per_tensor_scale(packed, scale, pts, 2)
    numerics = rq.RequantNumerics(scale_mode=scale_mode)
    a = torch.randn(e * r, k) * 0.02
    b = torch.randn(n, r * e)

    def merged(q, s, p, scaling):
        out = rq.requant_reference(
            {"qdata": q, "scale": s, "per_tensor_scale": p[:, None]},
            lora=(a, b, scaling, r),
            out_pts=p[:, None],
            e0=0,
            e1=e,
            numerics=numerics,
            with_atoms=False,
        )
        return out.qdata, out.scale, _dequant(out.qdata, out.scale, p)

    q0, s0, w0 = merged(wide_q, wide_s, wide_pts, 0.0)
    assert torch.equal(w0, _dequant(packed, scale, pts))
    assert torch.equal(q0, merged(packed, scale, pts, 0.0)[0])
    if scale_mode == "reuse":
        assert torch.equal(s0.view(torch.uint8), wide_s.view(torch.uint8))

    base = _dequant(packed, scale, pts)
    target = (
        (
            base
            + 2.0 * torch.bmm(b.reshape(n, r, e).permute(2, 0, 1), a.reshape(e, r, k))
        )
        .bfloat16()
        .float()
    )
    assert target.abs().amax() > 1.5 * base.abs().amax()
    err = lambda w: float((w - target).norm() / target.norm())
    assert err(merged(packed, scale, pts, 2.0)[2]) > 2 * err(
        merged(wide_q, wide_s, wide_pts, 2.0)[2]
    )

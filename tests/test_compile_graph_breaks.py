# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Graph breaks and recompiles of the public API under torch.compile, on the CPU harness
(``tests/compile_harness.py``; one subprocess per precision path)."""

import json
import os
import subprocess  # nosec B404
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
FULL_FINE_TUNE = ("bf16", "mxfp8", "fp8")
LORA = ("bf16_lora", "mxfp8_lora", "fp8_lora", "nvfp4_lora")
# The token count changes once (48 -> 40); automatic dynamic shapes absorb the rest
DYNAMIC_TOKENS_RECOMPILES = 1


def harness(*args):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(HERE.parent / "src"), env.get("PYTHONPATH")))
    )
    # the harness fakes the device; a visible GPU makes Dynamo query the fake's properties
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(  # nosec B603
        [sys.executable, str(HERE / "compile_harness.py"), *args, "--json"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("HARNESS_RESULT "):
            return json.loads(line[len("HARNESS_RESULT ") :])
    raise AssertionError(
        f"harness failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-8000:]}"
    )


def assert_clean(result, tolerance=0.0):
    assert result["graph_break_count"] == 0, result["break_reasons"]
    assert result["graph_count"] == 1
    assert result["compile_error"] is None
    assert result["logged_breaks"] == []
    assert result["recompiles"] <= DYNAMIC_TOKENS_RECOMPILES, result[
        "recompile_reasons"
    ]
    assert result["max_loss_diff"] <= tolerance
    assert result["max_grad_diff"] <= tolerance


@pytest.mark.parametrize("path", FULL_FINE_TUNE)
def test_forward_has_no_graph_breaks(path):
    assert_clean(harness(path))


@pytest.mark.parametrize("path", (*FULL_FINE_TUNE, *LORA))
def test_compiled_forwards_record_their_own_grad_mode(path):
    """Train, eval, train: a compiled eval forward records nothing and the next compiled
    training forward records every layer (its backward sizes the bounded pools from them)."""
    result = harness(path, "--records")
    assert result["pushed"] == result["expected"]


@pytest.mark.parametrize("path", LORA)
def test_registered_lora_layers_have_no_graph_breaks(path):
    assert_clean(harness(path, "--warm"))


@pytest.mark.parametrize("path", ("bf16_lora", "nvfp4_lora"))
def test_preregistered_lora_layers_have_no_graph_breaks(path):
    """More layers than the recompile limit, registered ahead of the first compiled forward."""
    assert_clean(harness(path, "--register", "--layers", "12"))


@pytest.mark.parametrize("path", ("bf16_lora", "nvfp4_lora"))
def test_lora_layers_first_seen_inside_compile_break_once(path):
    """Layers created lazily inside the compiled region register through one graph break each
    in the first step only, and registering never guards on the registry as a whole."""
    result = harness(path, "--no-fullgraph", "--layers", "4")
    breaks = result["breaks_after_step"]
    assert breaks[0] > 0 and breaks[-1] == breaks[0]
    assert not any("_LAYERS" in r for r in result["recompile_reasons"])
    assert result["max_loss_diff"] == 0.0
    assert result["max_grad_diff"] == 0.0


def test_fixed_layout_dispatch_has_no_graph_breaks():
    # the harness's local experts are torch ops traced into the graph, so not bitwise
    assert_clean(harness("dispatch"), tolerance=1e-5)


def test_compact_dispatch_breaks_only_at_the_host_count():
    """Compact mode reads the routed count on the host by design (one ``.item()`` per
    dispatch); that is its only graph break."""
    result = harness("dispatch_compact", "--no-fullgraph")
    assert result["break_reasons"]
    for reason in result["break_reasons"]:
        assert "item()" in reason["reason"]
        assert "peer_dispatch.py:" in reason["frames"][-1]


def test_bounded_dispatch_compiles_with_eager_numerics():
    """Bounded mode sizes its receive from host-side high-water counts, so it graph-breaks and
    recompiles by design; compiled, it must still match eager."""
    result = harness("dispatch_bounded", "--no-fullgraph")
    assert result["compile_error"] is None
    assert result["max_loss_diff"] <= 1e-5
    assert result["max_grad_diff"] <= 1e-5

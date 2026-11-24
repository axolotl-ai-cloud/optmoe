# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Environment switches."""

import os


def flag(name, default):
    """A ``0`` / ``1`` switch; any other value raises instead of picking a side."""
    value = os.getenv(name, "1" if default else "0")
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"

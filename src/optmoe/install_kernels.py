# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Install ``optmoe_deepgemm``, the compiled kernel extension the ``optmoe`` wheel does not carry.

    python -m optmoe.install_kernels            # prebuilt wheel from the GitHub release
    python -m optmoe.install_kernels --source   # build the release's sources
    python -m optmoe.install_kernels --source /path/to/optmoe

The extension links one torch minor version and one CUDA major version, so its wheels are
published per (Python, torch, CUDA) on the release page instead of PyPI. A source build needs
the CUDA toolkit, a C++20 compiler with ``<format>`` (GCC 13) and the ELF/DWARF headers
(``libelf-dev``, ``libdw-dev``).
"""

import argparse
import importlib.util
import os
import platform
import shutil
import subprocess  # nosec B404
import sys
import tempfile
from pathlib import Path
from urllib.parse import quote

REPOSITORY = "axolotl-ai-cloud/optmoe"


def release_version():
    """The installed ``optmoe`` release, without a local or development suffix."""
    import optmoe

    return optmoe.__version__.split("+")[0]


def build_tag():
    """``cu<major>.torch<major>.<minor>``: what the extension is built against."""
    import torch

    if torch.version.cuda is None:
        raise RuntimeError("optmoe_deepgemm needs a CUDA build of torch")
    cuda = torch.version.cuda.split(".")[0]
    major, minor = torch.__version__.split(".")[:2]
    return f"cu{cuda}.torch{major}.{minor}"


def wheel_name(version):
    python = f"cp{sys.version_info.major}{sys.version_info.minor}"
    machine = platform.machine()
    return (
        f"optmoe_deepgemm-{version}+{build_tag()}-{python}-{python}-linux_{machine}.whl"
    )


def wheel_url(version, repository=REPOSITORY):
    name = quote(wheel_name(version))
    return f"https://github.com/{repository}/releases/download/v{version}/{name}"


def _pip(*args):
    if importlib.util.find_spec("pip") is not None:
        command = [sys.executable, "-m", "pip", "install", *args]
    elif shutil.which("uv"):
        command = ["uv", "pip", "install", "--python", sys.executable, *args]
    else:
        raise RuntimeError("neither pip nor uv is available to install the extension")
    return subprocess.run(command, check=False).returncode  # nosec B603


def install_wheel(version, repository=REPOSITORY):
    return _pip("--no-deps", wheel_url(version, repository))


def install_source(version, checkout=None, repository=REPOSITORY):
    """Build ``vendor/deepgemm`` of ``checkout``, or of the release tag when not given."""
    with tempfile.TemporaryDirectory(prefix="optmoe-kernels-") as scratch:
        if checkout is None:
            checkout = Path(scratch) / "optmoe"
            subprocess.run(  # nosec B603 B607
                [
                    "git",
                    "clone",
                    "--depth=1",
                    f"--branch=v{version}",
                    f"https://github.com/{repository}.git",
                    str(checkout),
                ],
                check=True,
            )
        checkout = Path(checkout).resolve()
        subprocess.run(  # nosec B603
            [sys.executable, str(checkout / "tools/fetch_build_dependencies.py")],
            check=True,
        )
        # a reused object from before a header change fails at run time, so always build clean
        shutil.rmtree(checkout / "vendor/deepgemm/build", ignore_errors=True)
        os.environ.setdefault("OPTMOE_DEEPGEMM_VERSION", version)
        return _pip(
            "--no-build-isolation", "--no-deps", str(checkout / "vendor/deepgemm")
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--version", default=None, help="optmoe release (default: installed)"
    )
    parser.add_argument("--repository", default=REPOSITORY)
    parser.add_argument(
        "--source",
        nargs="?",
        const="",
        default=None,
        metavar="CHECKOUT",
        help="build from source: a local checkout, or the release tag when no path is given",
    )
    parser.add_argument(
        "--print-url", action="store_true", help="print the wheel URL and exit"
    )
    args = parser.parse_args(argv)
    version = args.version or release_version()
    if args.print_url:
        print(wheel_url(version, args.repository))
        return 0
    if args.source is not None:
        return install_source(version, args.source or None, args.repository)
    code = install_wheel(version, args.repository)
    if code:
        print(
            f"No prebuilt optmoe_deepgemm wheel installed for {build_tag()} on "
            f"Python {sys.version_info.major}.{sys.version_info.minor} "
            f"({wheel_url(version, args.repository)}).\n"
            "Build it from source instead: python -m optmoe.install_kernels --source",
            file=sys.stderr,
        )
    return code


if __name__ == "__main__":
    sys.exit(main())

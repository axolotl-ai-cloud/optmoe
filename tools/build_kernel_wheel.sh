#!/usr/bin/env bash
# Build the optmoe_deepgemm wheel inside a CUDA devel image for Rocky Linux 8 (no GPU needed).
#   tools/build_kernel_wheel.sh <python> <torch> <torch index, e.g. cu130> [version]
# Run from the repository root; the wheel lands in vendor/deepgemm/dist.
#
# Rocky 8 with gcc-toolset-13 gives the same floor as torch's own wheels (glibc 2.28) and links the
# parts of libstdc++ newer than the system's statically, so the wheel loads on Ubuntu 22.04 and
# later. A wheel built on Ubuntu 24.04 needs GLIBCXX_3.4.31 and fails to import on 22.04.
set -euo pipefail
python_version=$1
torch_version=$2
torch_index=$3
export OPTMOE_DEEPGEMM_VERSION=${4:-0.0.0.dev0}
export UV_NO_CACHE=1 CUDA_HOME=/usr/local/cuda MAX_JOBS=4

dnf install -y git curl ca-certificates gcc-toolset-13-gcc-c++ elfutils-devel elfutils-libelf-devel unzip binutils
# shellcheck disable=SC1091
source /opt/rh/gcc-toolset-13/enable
export CC=gcc CXX=g++
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python "$python_version" /opt/venv
uv pip install --python /opt/venv/bin/python \
  "torch==$torch_version" --index-url "https://download.pytorch.org/whl/$torch_index"
uv pip install --python /opt/venv/bin/python setuptools wheel packaging ninja
export PATH="/opt/venv/bin:$PATH"

git config --global --add safe.directory '*'
python tools/fetch_build_dependencies.py
cd vendor/deepgemm
rm -rf build dist
python setup.py bdist_wheel
ls -la dist

# the extension and its JIT headers must both be in the wheel
unzip -l dist/*.whl | grep -cE "include/deep_gemm/"
extension=$(unzip -Z1 dist/*.whl | grep -E "_C\..*\.so$")
unzip -p dist/*.whl "$extension" > /tmp/_C.so
symbols=$(objdump -T /tmp/_C.so | grep -oE "GLIBC(XX)?_[0-9.]+" | sort -uV)
echo "$symbols" | tail -n 4
# nothing newer than what Rocky 8 itself ships, or the wheel will not load on older systems
if echo "$symbols" | grep -qE "^GLIBCXX_3\.4\.(2[6-9]|[3-9][0-9])$|^GLIBC_2\.(29|[3-9][0-9])$"; then
  echo "the extension needs a newer glibc or libstdc++ than the Rocky 8 baseline" >&2
  exit 1
fi

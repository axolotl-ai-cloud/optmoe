#!/usr/bin/env bash
# Build the optmoe_deepgemm wheel inside a CUDA devel image (no GPU needed).
#   tools/build_kernel_wheel.sh <python> <torch> <torch index, e.g. cu130> [version]
# Run from the repository root; the wheel lands in vendor/deepgemm/dist.
set -euo pipefail
python_version=$1
torch_version=$2
torch_index=$3
export OPTMOE_DEEPGEMM_VERSION=${4:-0.0.0.dev0}
export DEBIAN_FRONTEND=noninteractive UV_NO_CACHE=1 CC=gcc-13 CXX=g++-13 CUDA_HOME=/usr/local/cuda MAX_JOBS=4

apt-get update
apt-get install -y --no-install-recommends git curl ca-certificates g++-13 libdw-dev libelf-dev unzip
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
unzip -l dist/*.whl | grep -cE "_C\..*\.so"
unzip -l dist/*.whl | grep -cE "include/deep_gemm/"

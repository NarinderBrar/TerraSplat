#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
export UV_CACHE_DIR="$PWD/.cache/uv"
uv venv --python 3.10 --allow-existing .venv-stage1
python3 scripts/prepare-skyfall-stage1.py
uv pip install --python .venv-stage1/bin/python torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-stage1/bin/python -r training/requirements-stage1.txt -c training/stage1-installed.txt
python3 scripts/install-cuda-toolchain.py
source scripts/stage1-env.sh
for extension in diff-gaussian-rasterization-depth simple-knn fused-ssim; do
  uv pip install --python .venv-stage1/bin/python --no-build-isolation ".external/skyfall-gs/submodules/$extension"
done
python scripts/check-stage1.py

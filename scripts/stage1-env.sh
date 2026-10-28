# Source this file from bash to select the workspace-only training toolchain.
TERRASPLAT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_HOME="$TERRASPLAT_ROOT/.toolchains/cuda-12.8"
export PATH="$TERRASPLAT_ROOT/.venv-stage1/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export UV_CACHE_DIR="$TERRASPLAT_ROOT/.cache/uv"
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=2
export OMP_NUM_THREADS=4
export MPLCONFIGDIR="$TERRASPLAT_ROOT/.cache/matplotlib"
export TORCH_HOME="$TERRASPLAT_ROOT/.cache/torch"
export XDG_CACHE_HOME="$TERRASPLAT_ROOT/.cache"

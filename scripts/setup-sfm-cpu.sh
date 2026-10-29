#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
export UV_CACHE_DIR="$PWD/.cache/uv"
TERRASPLAT_SFM_PREFIX="$PWD/.toolchains/sfm-cpu"

fetch_pinned() {
  local repository="$1" target="$2" revision="$3"
  if [[ ! -d "$target" ]]; then
    git clone --no-checkout "$repository" "$target"
    git -C "$target" checkout --detach "$revision"
  fi
  [[ "$(git -C "$target" rev-parse HEAD)" == "$revision" ]] || {
    echo "Unexpected source revision: $target" >&2
    exit 2
  }
  git -C "$target" diff --exit-code HEAD
}

mkdir -p .external
fetch_pinned https://github.com/ceres-solver/ceres-solver.git \
  .external/ceres-solver f68321e7de8929fbcdb95dd42877531e64f72f66
fetch_pinned https://github.com/Kai-46/ColmapForVisSat.git \
  .external/colmap-for-vissat 79a6d150714d0eb350a1b10e17bedf91877568a1

cmake -S .external/ceres-solver -B .external/ceres-solver/build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$TERRASPLAT_SFM_PREFIX" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_BENCHMARKS=OFF -DSUITESPARSE=OFF -DCXSPARSE=OFF -DLAPACK=OFF \
  -DSCHUR_SPECIALIZATIONS=OFF
cmake --build .external/ceres-solver/build --parallel 3
cmake --install .external/ceres-solver/build

cmake -S .external/colmap-for-vissat -B .external/colmap-for-vissat/build-cpu \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$TERRASPLAT_SFM_PREFIX" \
  -DCMAKE_PREFIX_PATH="$TERRASPLAT_SFM_PREFIX" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCERES_INCLUDE_DIRS="$TERRASPLAT_SFM_PREFIX/include" \
  -DCMAKE_POLICY_DEFAULT_CMP0167=OLD -DCUDA_ENABLED=OFF -DOPENGL_ENABLED=OFF -DOPENMP_ENABLED=OFF \
  '-DCMAKE_CXX_FLAGS=-include cstdio -include cstdint' \
  -DCGAL_ENABLED=OFF -DBOOST_STATIC=OFF -DTESTS_ENABLED=OFF
cmake --build .external/colmap-for-vissat/build-cpu --parallel 2
cmake --install .external/colmap-for-vissat/build-cpu

uv venv --python /usr/bin/python3 --allow-existing .venv-sfm
uv pip install --python .venv-sfm/bin/python -r sfm/requirements.txt -c sfm/installed.txt
uv pip freeze --python .venv-sfm/bin/python > "$TERRASPLAT_SFM_PREFIX/python-installed.txt"
python3 - <<'PY'
import hashlib,json,pathlib,subprocess
prefix=pathlib.Path('.toolchains/sfm-cpu')
binary=prefix/'bin/colmap'
with binary.open('rb') as stream:
    digest=hashlib.file_digest(stream,'sha256').hexdigest()
record={'colmap_commit':'79a6d150714d0eb350a1b10e17bedf91877568a1',
        'ceres_commit':'f68321e7de8929fbcdb95dd42877531e64f72f66',
        'executable_sha256':digest,'cuda':False,'opengl':False,'openmp':False,
        'compiler':subprocess.check_output(['c++','--version'],text=True).splitlines()[0],
        'cmake':subprocess.check_output(['cmake','--version'],text=True).splitlines()[0]}
(prefix/'runtime.json').write_text(json.dumps(record,indent=2)+'\n')
PY
"$TERRASPLAT_SFM_PREFIX/bin/colmap" -h

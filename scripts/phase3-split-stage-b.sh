#!/usr/bin/env bash
# Stage B: run ONLY after separate user approval of GPU training/evaluation.
# Run from the repository root. All outputs are new split-specific paths.
set -euo pipefail
export PYTHONPATH=
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
source scripts/stage1-env.sh
python - <<'PY'
import shutil
for directory in ('.','artifacts'):
    if shutil.disk_usage(directory).free < 20 * 1024**3:
        raise SystemExit(f'Less than 20 GiB free: {directory}')
PY
for block in west east; do
  python -m pipeline.phase2 "artifacts/imagery/${block}-19v/skyfall/phase2-manifest.json" --output "artifacts/imagery/${block}-19v/skyfall/preflight-gpu-report.json"
  python -m pipeline.training --scene "artifacts/imagery/${block}-19v/skyfall" --profile "configs/train-2015-${block}-19v-recipe.json" --output "artifacts/stage1/${block}-2015-19v-recipe-01"
  python -m pipeline.evaluation --run "artifacts/stage1/${block}-2015-19v-recipe-01"
  python -m pipeline.export --run "artifacts/stage1/${block}-2015-19v-recipe-01" --selection "artifacts/imagery/${block}-19v/selection.json"
  python -m pipeline.render_depth --run "artifacts/stage1/${block}-2015-19v-recipe-01" --output "artifacts/validation/${block}-2015-19v-recipe-01-depth"
  .venv-imagery/bin/python -m pipeline.dsm_validation --export "artifacts/stage1/${block}-2015-19v-recipe-01/export" --points "artifacts/validation/${block}-2015-19v-recipe-01-depth/depth-points.npz" --truth artifacts/reference/dfc2019-jax068 --grid configs/jax068-grid.json --cell "$block" --output "artifacts/validation/${block}-2015-19v-recipe-01-lidar"
done
.venv-imagery/bin/python -m pipeline.seam_check \
  --west-export artifacts/stage1/west-2015-19v-recipe-01/export \
  --east-export artifacts/stage1/east-2015-19v-recipe-01/export \
  --west-points artifacts/validation/west-2015-19v-recipe-01-depth/depth-points.npz \
  --east-points artifacts/validation/east-2015-19v-recipe-01-depth/depth-points.npz \
  --west-validation artifacts/validation/west-2015-19v-recipe-01-lidar \
  --east-validation artifacts/validation/east-2015-19v-recipe-01-lidar \
  --truth artifacts/reference/dfc2019-jax068 --grid configs/jax068-grid.json \
  --decision configs/phase3-split-decision.json --output artifacts/validation/phase3-split-seam-check.json
# set -e stops here if the pre-registered seam/MAE/ownership rule fails.
for block in west east; do
  .venv-imagery/bin/python -m pipeline.tiling --export "artifacts/stage1/${block}-2015-19v-recipe-01/export" --grid configs/jax068-grid.json --cell "$block" --output "artifacts/tiles/${block}-2015-19v-recipe-01-r1"
  node scripts/encode-tiles.mjs "artifacts/tiles/${block}-2015-19v-recipe-01-r1"
  .venv-sfm/bin/python scripts/phase3-split-evidence.py --block "$block"
  .venv-sfm/bin/python -m pipeline.publish --config "configs/releases/jax068-2015-${block}-19v.json" --releases artifacts/releases-phase3-split
done
# New releases are inactive and isolated from artifacts/releases and its catalogue.

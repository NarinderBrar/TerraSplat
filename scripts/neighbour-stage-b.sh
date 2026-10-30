#!/usr/bin/env bash
# Stage B: requires separate user authorization for GPU training/evaluation.
# PRECONDITION 1: artifacts/reference/3dep-jax068-east/qualification.json
# must exist with the JSON boolean "accepted": true, qualifying the stable key.
# PRECONDITION 2: configs/neighbour-decision.json must exist, pre-registered.
# Deferred step 5 must also implement scripts/neighbour-seam-check.py with the
# CLI below, emitting decision.passed and exiting nonzero on fail/inconclusive.
# It must verify source hashes, same-key/cell comparisons, residual seam rules,
# minimum evidence and ownership. The split checker is NOT a substitute.
# Run from repository root; all new outputs are neighbour-specific. No resume.
set -euo pipefail
export PYTHONPATH=
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
# Fail before environment setup, CUDA probing or any output-writing operation.
.venv-imagery/bin/python -m pipeline.neighbour_preparation stage-b-preflight
source scripts/stage1-env.sh
run=artifacts/stage1/eastn-2015-19v-recipe-01
central_export=artifacts/stage1/pilot-2015-19v-recipe-01/export-r2
truth=artifacts/reference/3dep-jax068-east/stable
grid=configs/jax068-grid-neighbour.json
python -m pipeline.phase2 artifacts/imagery/eastn-19v/skyfall/phase2-manifest.json --output artifacts/imagery/eastn-19v/skyfall/preflight-gpu-report.json
python -m pipeline.training --scene artifacts/imagery/eastn-19v/skyfall --profile configs/train-2015-eastn-19v-recipe.json --output "$run"
python -m pipeline.evaluation --run "$run"
python -m pipeline.export --run "$run" --selection artifacts/imagery/eastn-19v/selection.json
python -m pipeline.render_depth --run "$run" --output artifacts/validation/eastn-2015-19v-recipe-01-depth
# Read the existing central export and depth points; write only a new validation.
.venv-imagery/bin/python -m pipeline.dsm_validation --export "$central_export" --points artifacts/validation/pilot-2015-19v-recipe-01-depth-r2/depth-points.npz --truth "$truth" --grid "$grid" --cell centre --output artifacts/validation/neighbour-centre-2015-19v-lidar
.venv-imagery/bin/python -m pipeline.dsm_validation --export "$run/export" --points artifacts/validation/eastn-2015-19v-recipe-01-depth/depth-points.npz --truth "$truth" --grid "$grid" --cell east --output artifacts/validation/eastn-2015-19v-recipe-01-lidar
.venv-imagery/bin/python scripts/neighbour-seam-check.py \
  --centre-export "$central_export" --east-export "$run/export" \
  --centre-points artifacts/validation/pilot-2015-19v-recipe-01-depth-r2/depth-points.npz \
  --east-points artifacts/validation/eastn-2015-19v-recipe-01-depth/depth-points.npz \
  --centre-validation artifacts/validation/neighbour-centre-2015-19v-lidar \
  --east-validation artifacts/validation/eastn-2015-19v-recipe-01-lidar \
  --truth "$truth" --grid "$grid" --decision configs/neighbour-decision.json \
  --output artifacts/validation/neighbour-seam-check.json
# Fail closed even if the future checker incorrectly exits zero with failed rules.
python - <<'PY'
import json
from pathlib import Path
report = json.loads(Path('artifacts/validation/neighbour-seam-check.json').read_text())
if report['decision']['passed'] is not True:
    raise SystemExit('Neighbour rules failed or evidence is inconclusive')
PY
.venv-imagery/bin/python -m pipeline.tiling --export "$run/export" --grid "$grid" --cell east --output artifacts/tiles/eastn-2015-19v-recipe-01-r1
node scripts/encode-tiles.mjs artifacts/tiles/eastn-2015-19v-recipe-01-r1
.venv-imagery/bin/python scripts/neighbour-evidence.py
# Publication is isolated and inactive: no --activate, no live catalogue writes.
.venv-imagery/bin/python -m pipeline.publish --config configs/releases/jax068-2015-eastn-19v.json --releases artifacts/releases-neighbour

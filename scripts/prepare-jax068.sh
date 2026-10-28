#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
TERRASPLAT_RUN="${1:-artifacts/imagery/pilot-$(date +%Y%m%d-%H%M%S)}"
if [[ -e "$TERRASPLAT_RUN" ]]; then
  echo "Use a new run directory; existing outputs were left unchanged." >&2
  exit 2
fi
export UV_CACHE_DIR="$PWD/.cache/uv"
export OPENBLAS_NUM_THREADS=2
uv venv --python /usr/bin/python3 --allow-existing .venv-imagery
uv pip install --python .venv-imagery/bin/python -r imagery/requirements.txt -c imagery/installed.txt
python3 scripts/prepare-satellitesfm.py
.venv-imagery/bin/python -m pipeline.imagery catalogue --output "$TERRASPLAT_RUN/catalogue.json"
.venv-imagery/bin/python -m pipeline.acquisitions scan "$TERRASPLAT_RUN/catalogue.json" --block configs/jax068-block.json --output "$TERRASPLAT_RUN/acquisitions.json"
.venv-imagery/bin/python -m pipeline.acquisitions select "$TERRASPLAT_RUN/acquisitions.json" --max-span-days 14 --output "$TERRASPLAT_RUN/selection.json"
.venv-imagery/bin/python - "$TERRASPLAT_RUN/selection.json" <<'PY'
import json,sys
from pathlib import Path
selected=json.loads(Path(sys.argv[1]).read_text())
approved=json.loads(Path('configs/jax068-pilot.json').read_text())
for role in ('train','holdout'):
    actual={v['catalog_id'] for v in selected[role]}
    if actual!=set(approved[f'{role}_catalog_ids']):
        raise SystemExit('Candidate differs from the approved pilot; source download was not started')
PY
.venv-imagery/bin/python -m pipeline.sources "$TERRASPLAT_RUN/selection.json" --output "$TERRASPLAT_RUN/sources.json"
.venv-imagery/bin/python -m pipeline.block_inputs "$TERRASPLAT_RUN/selection.json" "$TERRASPLAT_RUN/sources.json" --output "$TERRASPLAT_RUN/inputs"
.venv-imagery/bin/python -m pipeline.satellite_sfm "$TERRASPLAT_RUN/selection.json" --output "$TERRASPLAT_RUN/calibration"
echo "Prepared inputs and initial cameras: $TERRASPLAT_RUN"

# Phase 3 split preparation — 2026-10-30

Stage A prepares west/east scenes only. No training, evaluation rendering, CUDA
job, imagery download or publication is authorized in this stage. Existing pilot
outputs, release configurations, runs, gates and `apps/web/` are preserved.
No commit, push or Git index/worktree mutation commands were used.

Initial free space: workspace volume approximately 78 GiB; artifacts volume
approximately 62 GiB. Both exceed the 20 GiB stop threshold.

The older status paragraphs in `docs/PLAN_V2.md` still describe the earlier
19-view export blocker. The measured recipe result, findings and releases docs
establish the subsequent 1.359 m registered MAE baseline; no old result is edited.

## Geometry and truth coverage

Anchor: longitude −81.66376054127814°, latitude 30.34881972282938°.
The shared grid uses ellipsoidal height 0 m to complete the anchor definition;
the source block config specifies no anchor altitude. Block centres are derived
by transforming shared ENU (−125, 0, 0) / (+125, 0, 0) through ECEF to WGS84.
Only the resulting longitude/latitude enter the block configs. The established
calibration code still chooses each scene's own observer near its centre at
ellipsoidal height −70 m; none of its settings or conventions is changed.

| Block | Centre longitude | Centre latitude | Shared east (m) | Shared north (m) |
| --- | ---: | ---: | --- | --- |
| west | −81.66506063308124 | 30.348819716365696 | [−250, 0) | [−125, 125) |
| east | −81.66246044947502 | 30.34881971636569 | [0, 250) | [−125, 125) |

Each has side 250 m, halo 20 m, and the unchanged [−50, 300] m ellipsoidal
crop height bracket. Image cropping retains the existing UTM perimeter plus
halo; delivery ownership is defined in the shared ENU frame.

Own-frame boundary mismatch at ellipsoid height 0: **2.866 mm maximum 3D**,
almost entirely east-west. Using actual calibration anchors and 1,001 samples
along the boundary at each of −50, 0 and 300 m gives **14.615 mm maximum 3D**.
North discrepancy is below 0.00018 mm and vertical discrepancy below 0.000002 mm.
These are negligible for image sampling but violate a strict zero-width ownership
gap/overlap rule. Tiling therefore transforms each centre's full XYZ through
ECEF to the shared grid and applies half-open cells; east owns the seam.
Each model, camera and placed export retains its own ENU coordinates.
Detailed measurements: `training/phase3-split-grid.json`.

DFC2019 truth is 512×512 at 0.5 m. Testing pixel centres in the ownership frame
produces the following **geometric eligibility counts**, before any prediction:

| Block | Truth pixels in cell | Labelled eligible pixels | Fraction of block area covered | Fraction of truth grid |
| --- | ---: | ---: | ---: | ---: |
| west | 127,148 | 119,148 | 0.508592 | 0.485031128 |
| east | 128,768 | 122,773 | 0.515072 | 0.491210938 |

The lidar is UTM-aligned and the grid ENU-aligned; the pair covers most of the
truth tile's central north-south strip, not exactly two equal raster halves.
The shared seam is inside the truth, with a minimum 1.904 m margin to the raster edge across 3,003 boundary/height samples. The remaining truth pixels are outside
the cells' north/south bounds. The two ownership masks intersect at zero pixels.
`training/phase3-split-coverage.json` uses truth as a synthetic perfect prediction
to quantify the eligible pixels; its evaluated counts are **not trained results**.
Actual evaluated pixel counts and labelled completeness remain unavailable until
rendered-depth DSMs exist. `dsm_validation` now reports these explicitly, including
block-area coverage and truth-grid coverage, and rejects zero overlap.

The old validator did mask finite predictions, so it did not assume every truth
pixel was predicted. Its diagonal-corner UTM bounding rectangle approximated the
ENU square and obscured partial coverage. It now tests every truth pixel centre
in the actual ENU ownership frame. A synthetic half-overlap regression test recovers
the exact known registration offset with 8,192 evaluated pixels and 50% coverage.

## Stage A commands

All commands run from the repository root. `--offline` forbids network access,
including when a cached source or sidecar is missing. Sources are hash-verified
in the existing shared `artifacts/imagery/raw` cache; no source is copied or downloaded.
The two selections preserve the exact original acquisition lists; May 2 stays
out of calibration fitting, SfM, training and pseudo views (its RPC camera is
prepared independently for later evaluation only).

```bash
export PYTHONPATH=
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
for block in west east; do
  .venv-imagery/bin/python -m pipeline.acquisitions scan artifacts/imagery/catalogue.json --block "configs/jax068-${block}-block.json" --offline --output "artifacts/imagery/${block}-19v/acquisitions.json"
  .venv-imagery/bin/python -m pipeline.acquisitions select "artifacts/imagery/${block}-19v/acquisitions.json" --approved "configs/jax068-${block}-19v.json" --output "artifacts/imagery/${block}-19v/selection.json"
  .venv-imagery/bin/python -m pipeline.sources "artifacts/imagery/${block}-19v/selection.json" --offline --max-total-gib 30 --output "artifacts/imagery/${block}-19v/sources.json"
  .venv-imagery/bin/python -m pipeline.block_inputs "artifacts/imagery/${block}-19v/selection.json" "artifacts/imagery/${block}-19v/sources.json" --output "artifacts/imagery/${block}-19v/inputs"
  .venv-imagery/bin/python -m pipeline.satellite_sfm "artifacts/imagery/${block}-19v/selection.json" --output "artifacts/imagery/${block}-19v/calibration"
  .venv-sfm/bin/python -m pipeline.reconstruct --inputs "artifacts/imagery/${block}-19v/inputs" --calibration "artifacts/imagery/${block}-19v/calibration" --approved "configs/jax068-${block}-19v.json" --output "artifacts/imagery/${block}-19v/sfm-serial"
  .venv-sfm/bin/python -m pipeline.skyfall_inputs --reconstruction "artifacts/imagery/${block}-19v/sfm-serial" --inputs "artifacts/imagery/${block}-19v/inputs" --approved "configs/jax068-${block}-19v.json" --output "artifacts/imagery/${block}-19v/skyfall"
  source scripts/stage1-env.sh
  python scripts/check-skyfall-inputs.py --scene "artifacts/imagery/${block}-19v/skyfall"
  python -m pipeline.phase2 "artifacts/imagery/${block}-19v/skyfall/phase2-manifest.json" --skip-environment --output "artifacts/imagery/${block}-19v/skyfall/preflight-report.json"
done
```

Calibration, reconstruction and conversion require new output directories;
do not blindly rerun completed commands. The new CLI `--skip-environment` selects
the existing `preflight(..., check_environment=False)` CPU audit. It leaves the
normal full preflight and training gates unchanged. The deferred CUDA probe is
first in the separately approved Stage B script.

## Pre-registered decision

`configs/phase3-split-decision.json` was written before any split training:

- Each block's registered all-labelled MAE ≤ **1.659 m** (1.359 + the existing
  0.3 m regression budget).
- Each block's ±10 m seam-band MAE ≤ its >10 m interior MAE + **0.5 m**.
  This budget is comparable to the prior 3→5 view gain and smaller than the
  1.458 m gain from 5-view basic to 19-view recipe. The 0.5 m horizontal NCC
  sampling step is context, not a claimed vertical error bound.
- Absolute difference of raw ellipsoidal height medians in the two 10 m strips:
  **≤0.5 m ground**, **≤1.0 m all pixels**. Ground budget reflects the prior
  pilot's approximately 0.2–0.6 m vertical uncertainty; the all-pixel budget allows
  composition differences but remains below the baseline 1.359 m shape error.
  Lidar's own strip step and registered/residual steps are also reported.
- **Zero** centres owned by both crops; **zero** width gap and overlap in the
  analytic ownership partition. Counts in ±1 m are reported per block.
- Empty comparison subsets fail/inconclusive. No replicated same-input run
  variance exists: these are explicit policy tolerances, not confidence bounds.
  Natural scene differences can fail the conservative raw-strip step rule;
  thresholds must not be retuned from trained results.

The checker verifies export, depth, validation, truth and ownership identities,
plus validation raster hashes. Height continuity uses raw heights so independent
vertical registration cannot conceal a step. Registered MAE measures surface
shape and is reported separately from placement. Centre ownership does not
establish disjoint Gaussian kernel support or visual mixed-LOD continuity.

## CPU execution notes

The initial reconstructions were run concurrently in `west-19v/sfm` and
`east-19v/sfm`. Both feature extraction stages passed (86.05 / 87.00 seconds),
but exhaustive matching reached the unchanged 900-second per-command timeout.
Neither failed attempt produced sparse points or reprojection statistics.
The failed reports and logs are preserved. A sequential retry uses new
`sfm-serial` directories with identical SfM arguments and the same timeout;
the Stage A commands above show these retry paths. No gate or profile was relaxed.

## Per-block preparation status

West: acquisition selection passed (19 train / 1 holdout), all 40 cached source
products reused, RGB inputs prepared, RPC calibration passed. Sequential SfM
passed with **23,157 points**, **0.421855 px median** reprojection error (79,385
observations; p95 1.502907 px). Matching took 607.76 seconds. Skyfall conversion,
the real loader check and CPU phase2 preflight passed. CUDA environment probing
is deferred to Stage B, and no training was run.

East: all preparation steps passed with the same 19/1 acquisition split and all
40 source products reused. Sequential SfM produced **24,009 points**,
**0.404656 px median** reprojection error (90,306 observations; p95 1.471334 px).
Matching took 609.66 seconds. Skyfall conversion, the real loader check and CPU
phase2 preflight passed. CUDA environment probing remains deferred.

| Stage | West | East |
| --- | --- | --- |
| Offline acquisition scan + approved selection | Passed, 19 + 1 | Passed, 19 + 1 |
| Sources | 40 cache products verified/reused | 40 cache products verified/reused |
| block_inputs | Prepared for review | Prepared for review |
| satellite_sfm RPC calibration | Passed | Passed |
| reconstruct (sequential retry) | 23,157 points; 0.421855 px median | 24,009 points; 0.404656 px median |
| skyfall_inputs | Prepared | Prepared |
| check-skyfall-inputs | Passed | Passed |
| phase2 CPU preflight | Passed; environment_checked=false | Passed; environment_checked=false |
| Full CUDA environment preflight | Deferred to approved Stage B | Deferred to approved Stage B |
| Training / evaluation / export / rendered depth / publication | Not run | Not run |

No CPU preparation step remains blocked. The failed concurrent attempts remain
in the original `sfm` folders; the successful reports are in `sfm-serial`.
All subsequent scenes use those successful reconstructions.

## Stage B (not executed)

Requires separate user approval: about **84.2 minutes** for two recipe trainings
from the existing 42.1-minute measurement, plus evaluation/export/rendering.
Exact commands are in the Phase 3 section of `tasks.md` and
[`scripts/phase3-split-stage-b.sh`](../../scripts/phase3-split-stage-b.sh).
Run `bash scripts/phase3-split-stage-b.sh` only after approval.

Order: full GPU-environment preflight → training → holdout evaluation → export →
render_depth → registered dsm_validation for each block → paired seam_check →
shared-grid tiling → SPZ encoding → measured publication evidence → publish.
A failed pre-registered decision stops the script before tiling/publication.
New releases publish inactive into **artifacts/releases-phase3-split**, preserving
the current catalogue. `scripts/phase3-split-evidence.py` gathers measured values;
it never supplies placeholder quality values. New per-block release configs are
prepared in `configs/releases/`.

## Verification and file inventory

**64 tests passed** using:

```bash
export PYTHONPATH=
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
.venv-imagery/bin/python -m pytest tests/sfm/test_seam_check.py tests/sfm/test_tiling.py tests/sfm/test_publish.py tests/imagery/test_dsm_validation.py tests/imagery/test_sources.py tests/imagery/test_acquisitions.py -q
bash -n scripts/phase3-split-stage-b.sh
git diff --check
```

Tests cover half-open ownership at the seam, gaps/overlaps, independently tilted
ENU frames, shared-grid tiling and publication footprints, end-to-end synthetic
export/depth/DSM/seam reporting, hash mismatch rejection, known height steps,
missing samples, half-truth registration/coverage, unchanged acquisition roles
and profiles, offline cache misses, and measured publication evidence collection.
Rasterio's expected warnings reflect the DFC/Sat-NeRF convention of separate
DSM.txt grid metadata rather than TIFF geotransforms.

The complete added/changed file list, per-stage report paths and hashes, initial
failed attempts, final sparse statistics and free-space measurements are in
`training/phase3-split-preparation.json`. `tasks.md` is a local ignored file and
has also been updated. All changes remain uncommitted.

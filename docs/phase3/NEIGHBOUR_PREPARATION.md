# Neighbour preparation independent of the answer key — 2026-10-30

The user explicitly authorized steps 4, 6 and the guarded step 7 while the 3DEP
answer key remains on hold. The qualification results below are historical and
unchanged. No answer-key output, `pipeline/lidar_reference.py`, qualification
threshold or `configs/neighbour-decision.json` was changed. No training, GPU job,
new imagery/lidar download or publication ran. Existing releases, live catalogue,
runs, split outputs/rules and `apps/web/` were not modified by this work.

## Step 4: grid and central ownership

`configs/jax068-grid-neighbour.json` uses the original anchor and ENU→ECEF
matrix. Half-open cells are centre east [-125,125), east [125,375), both north
[-125,125). The east boundary belongs to east; the analytic partition has zero
width gap/overlap. Eastn's centre is shared ENU (250,0,0), longitude
−81.66116035767325°, latitude 30.348819696974616°. Its image crop retains the
existing 250 m UTM square plus 20 m halo and [-50,300] m height bracket.

Measured the exact full-resolution central release export (`export-r2`),
checking its manifest/PLY hashes and count against the delivered tile plan.
Reproducible read-only audit:

```bash
export PYTHONPATH=
.venv-imagery/bin/python -m pipeline.neighbour_preparation ownership --output training/neighbour-ownership.json
```

Use a new output path when repeating. Evidence: `training/neighbour-ownership.json`.

| Central release / proposed shared crop | Centres |
| --- | ---: |
| Existing closed own-frame crop | 66,606 |
| Exactly on east=125 (own frame / shared frame) | 0 / 0 |
| Within 1 m of east=125 (own frame / shared frame) | 176 / 176 |
| Existing owned centres inside neighbour half-open cell | 0 |
| Proposed shared centre crop | 66,604 |
| Removed from existing / added from original export | 3 / 1 |
| Proposed doubled centres | 0 |

Maximum shared east of existing owned centres is **124.995604034 m**; the
closest lies **0.004395966 m** west of the seam. There is no measured doubled
centre at this seam, but the closed crop does not establish a half-open partition.
Over the owned centres’ actual Up range (23.267–103.050 m), the closed crop’s
east edge maps to shared east 124.998070552–124.998070620 m, a **1.92938–1.92945 mm
geometric gap** to the neighbour boundary. These are affine edge extrema at
its corners, not the nearest centre distance. It uses a slightly different
calibrated observer (height −70 m), so merely
changing ≤ to < in its own frame does not implement the shared-frame fix.

**Proposal only:** re-tile the unchanged central `export-r2` using this grid's
`centre` cell, encode as a new release such as `jax068-2015-19v-neighbour-r1`,
and retain the old release. This would own 66,604 centres. No central tile/release
was created or published. `partition_check` now accepts centre/east as well as
west/east, and the tiling/validation CLIs accept `centre`; split behavior is
covered by the existing regression tests. Counts describe full-resolution
centres, not LOD parent centroids, Gaussian kernel support, or visual continuity.

## Step 6: CPU eastn preparation

The approved selection preserves the exact ordered 19 training ids and May 2
holdout from `jax068-pilot-19v.json`. The recipe differs from
`train-2015-pilot-19v-recipe.json` **only in id**; all numerical settings, notes,
model path and timeouts are identical. May 2 is prepared independently as the
existing evaluation-only camera/image, excluded from fitting, COLMAP database,
training and pseudo views; no holdout image/metric informed a choice.

All commands run offline against existing sources:

```bash
export PYTHONPATH=
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
.venv-imagery/bin/python -m pipeline.acquisitions scan artifacts/imagery/catalogue.json --block configs/jax068-eastn-block.json --offline --output artifacts/imagery/eastn-19v/acquisitions.json
.venv-imagery/bin/python -m pipeline.acquisitions select artifacts/imagery/eastn-19v/acquisitions.json --approved configs/jax068-eastn-19v.json --output artifacts/imagery/eastn-19v/selection.json
.venv-imagery/bin/python -m pipeline.sources artifacts/imagery/eastn-19v/selection.json --offline --max-total-gib 30 --output artifacts/imagery/eastn-19v/sources.json
.venv-imagery/bin/python -m pipeline.block_inputs artifacts/imagery/eastn-19v/selection.json artifacts/imagery/eastn-19v/sources.json --output artifacts/imagery/eastn-19v/inputs
.venv-imagery/bin/python -m pipeline.satellite_sfm artifacts/imagery/eastn-19v/selection.json --output artifacts/imagery/eastn-19v/calibration
.venv-sfm/bin/python -m pipeline.reconstruct --inputs artifacts/imagery/eastn-19v/inputs --calibration artifacts/imagery/eastn-19v/calibration --approved configs/jax068-eastn-19v.json --output artifacts/imagery/eastn-19v/sfm
.venv-sfm/bin/python -m pipeline.skyfall_inputs --reconstruction artifacts/imagery/eastn-19v/sfm --inputs artifacts/imagery/eastn-19v/inputs --approved configs/jax068-eastn-19v.json --output artifacts/imagery/eastn-19v/skyfall
source scripts/stage1-env.sh
python scripts/check-skyfall-inputs.py --scene artifacts/imagery/eastn-19v/skyfall
python -m pipeline.phase2 artifacts/imagery/eastn-19v/skyfall/phase2-manifest.json --skip-environment --output artifacts/imagery/eastn-19v/skyfall/preflight-report.json
```

Do not rerun completed commands into existing output directories.

Coverage: all **20 acquisitions / 40 PAN+MSI products** fully cover the neighbour
block plus halo according to the existing `rpc_window` procedure: 17 samples
per edge of the 290 m UTM footprint, at −50, 125 and 300 m ellipsoidal heights,
with an additional 8 px padding. Minimum remaining source margin: **103 px**.
`training/neighbour-coverage.json` records every image size/window and role.
All 40 source hashes verified and products reused; the legacy sources report
status says `downloaded` but every product has `reused: true`, with no new
transfer. All RGB crops have valid fraction 1.0. Coverage of the holdout is source
QC only. This is the established sampled RPC coverage check, not a claim of an
analytic bound on a nonlinear RPC between samples.

| Stage | Eastn status |
| --- | --- |
| Block config / approved selection / fixed recipe | Prepared; original 19/1 roles and recipe preserved |
| Offline acquisition scan + select | Passed, 19 train + 1 holdout |
| Offline source verification | Passed, all 40 cached PAN/MSI products reused |
| block_inputs | Prepared, all 20 valid fractions 1.0 |
| satellite_sfm | Passed; holdout RPC-only |
| reconstruct | **24,461 points; 0.470499 px median**, 92,095 observations; p95 1.514585 px |
| skyfall_inputs | Prepared, 24,461 points |
| check-skyfall-inputs | Passed, 19 train / 1 holdout |
| CPU preflight --skip-environment | Passed, environment_checked=false |
| GPU environment / training / evaluation / publication | Not run |

Matching took **866.944 seconds**, inside the unchanged 900-second timeout;
feature extraction took 112.166 seconds. Both triangulation and bundle-adjustment
passes completed normally. A direct read-only SQLite audit confirms exactly the
approved 19 training image names and no May 2 image in the COLMAP database.
No retry or gate/parameter relaxation was needed. The source/conversion reports
and their hashes, exact commands, statuses and free space are recorded in
`training/neighbour-preparation.json`. No CPU preparation step remains blocked.

## Step 7: Stage B script, staged and deliberately blocked

`scripts/neighbour-stage-b.sh` documents and enforces both prerequisites before
sourcing the GPU environment, probing CUDA or writing any output:

1. `artifacts/reference/3dep-jax068-east/qualification.json` exists with the
   JSON boolean **`"accepted": true`**. False, missing, null, numeric 1 and string
   "true" are rejected. The stable truth payload must also exist.
2. `configs/neighbour-decision.json` exists and parses as a JSON object.

The rejected qualification is unchanged and the decision file was deliberately
not written. The guard was safely exercised: `bash scripts/neighbour-stage-b.sh`
exits **2**, `Answer key is not qualified (accepted must be true):
artifacts/reference/3dep-jax068-east/qualification.json`. No GPU work starts.

There is an additional required dependency from deferred **step 5**:
`scripts/neighbour-seam-check.py`. It is intentionally **not implemented** here
because new rules/computations are outside the authorized steps. The script
specifies its future CLI, and the startup guard refuses if it is missing even
after the two JSON prerequisites exist. The old split checker cannot enforce
same-key neighbour-relative MAE, residual seam limits or minimum evidence, so it
is never called. This Stage B is staged, not executable end to end until step 5
is completed and separately reviewed/authorized for GPU execution.

The command chain is full environment preflight → neighbour recipe training →
holdout evaluation → export → rendered depth → new per-cell validation of the
neighbour and existing central export against the **same stable 3DEP key** →
future neighbour seam/rule checker → explicit boolean pass check → shared-grid
neighbour tiling → SPZ encoding → measured evidence → inactive publication into
`artifacts/releases-neighbour`. Central export/depth are read-only; its new
validation is written to `artifacts/validation/neighbour-centre-2015-19v-lidar`.
No existing central run is re-exported or re-rendered. The release config and
evidence collector are prepared, with no placeholder quality values. A failed
or inconclusive rule report stops before tiling/publication. No central re-tile
is part of this script; the proposal above remains unpublished.

## Verification, inventory and exclusions

**86 tests passed** (expected rasterio warnings for the existing separate text
georeferencing convention). New tests cover fixed acquisition roles and exact
recipe identity, centre position, closed-edge doubled-centre synthetic case,
frame translation, half-open neighbour partition and tiny gap/overlap rejection,
actual-grid centre/east tiling and publication footprints, missing/rejected/
non-boolean qualification, missing decision/checker/truth, low disk, shell
refusal before environment/output writes, and measured publication evidence with
failed-rule and incorrect-holdout rejection. Existing split tests still pass.

```bash
export PYTHONPATH=
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
.venv-imagery/bin/python -m pytest tests/imagery/test_neighbour_preparation.py tests/sfm/test_seam_check.py tests/sfm/test_tiling.py tests/sfm/test_publish.py tests/imagery/test_dsm_validation.py tests/imagery/test_sources.py tests/imagery/test_acquisitions.py -q
bash -n scripts/neighbour-stage-b.sh
git diff --check -- pipeline docs/phase3/NEIGHBOUR_PREPARATION.md configs scripts tests training
```

The first ownership command used `.venv-sfm`, which lacks `plyfile`; the audit
succeeded with `.venv-imagery`. This was an environment-package mismatch, not a
sandbox denial. No CPU preparation command required escalation.

Added files: `configs/jax068-grid-neighbour.json`, `configs/jax068-eastn-block.json`,
`configs/jax068-eastn-19v.json`, `configs/train-2015-eastn-19v-recipe.json`,
`configs/releases/jax068-2015-eastn-19v.json`, `pipeline/neighbour_preparation.py`,
`scripts/neighbour-stage-b.sh`, `scripts/neighbour-evidence.py`,
`tests/imagery/test_neighbour_preparation.py`, `training/neighbour-ownership.json`,
`training/neighbour-coverage.json`, `training/neighbour-preparation.json`.
Updated: `pipeline/ownership.py`, `pipeline/tiling.py`,
`pipeline/dsm_validation.py`, this document and local ignored `tasks.md`.
Generated artifacts are confined to `artifacts/imagery/eastn-19v/`.

Not done: answer-key reconsideration/conversion/qualification, step 5 decision
file or new rule computations, GPU preflight, training, evaluation, export,
rendered depth, model grading, central re-tiling or any publication. No sandbox
escalation was requested. Changes remain uncommitted; no index-changing Git
commands, commits or pushes were used. Initial free space: workspace about
77 GiB, artifacts about 60 GiB; final free space is 76.4 GiB / 59.9 GiB respectively, above the 20 GiB stop threshold.

---

# Prior run: true-neighbour 3DEP qualification — 2026-10-30

**Stopped at step 3: answer key rejected.** No neighbour training, GPU jobs,
imagery downloads, CPU imagery/SfM preparation or publication ran. Steps 4–8
remain unexecuted as required by the fixed qualification gate. No existing
releases, runs, split outputs/rules, live catalogue or web files were modified.
Changes are uncommitted. Free space initially: workspace 77 GiB, artifacts
61 GiB, both above the 20 GiB minimum.

## Datum evidence and transformation

Read the task's required files, including the binding lessons in
`SPLIT_RESULT.md`: future comparisons must use the same answer key/procedure on
each cell, and seam continuity must use residuals beyond the truth's own step.
Those later rules were not implemented because this key failed qualification.

2018 official per-tile XML under `artifacts/reference/usgs-3dep/metadata/`
documents NAVD88, NAD83(2011)/Florida East and US survey feet for the delivered
LAZ. The downloaded EPT points already use EPSG:3857 XY and **metres** Z;
no feet-to-metres scaling was applied again. Flight dates: November 28–December
18, 2018. Input NPZ hashes were verified against their read-only manifests.

2018 conversion, with PROJ networking disabled:

1. EPSG:3857 → WGS84 longitude/latitude; retain those horizontal positions for
   EPSG:32617 output. EPT's horizontal reprojection has already happened.
2. Inverse of the explicit NAD83(2011) → WGS84(G1762) operation locates NAD83
   longitude/latitude for geoid sampling. Iterate three times with the computed
   ellipsoid height to resolve the negligible height dependence of latitude.
3. NAVD88 → NAD83(2011) ellipsoid: `h = H + N`, using explicit PROJ
   `vgridshift`, multiplier +1, and GEOID18
   `/home/narinder/.local/share/proj/us_noaa_g2018u0.tif`.
4. NAD83(2011) → ITRF2008 → WGS84(G1762), pyproj EPSG:6319 → EPSG:7665,
   inverse EPSG operation 7807 and inverse WGS84(G1762) to ITRF2008 (1).
   This is the full time-dependent Helmert, evaluated explicitly at **2010.0**,
   the NAD83(2011) reference epoch supplied by GEOID18, rather than silently
   defaulting to the transformation's 1997 parameter epoch or the flight date.
   EPSG:32617 carries the resulting WGS84 ellipsoidal heights.

The exact pyproj pipelines, full operation PROJJSON, epoch and input/document
hashes are in
`artifacts/reference/3dep-jax068-east/qualification.json`.
GEOID18 SHA256:
`fa9a407ac7ee3f5a3694008e4bcd09ce9cc250452f0c3b11700a4960340abce2`.
No additional grids were needed. Pyproj's direct EPSG:6319 → EPSG:4979 no-op
was not used in place of this realization-specific transformation.

At the anchor, NAVD88 zero is **−28.263635 m** NAD83(2011) ellipsoidal and
**−29.751054 m** WGS84(G1762) ellipsoidal; the frame height change is
**−1.487419 m**. No offsets were fitted to DFC2019.
NGS documents GEOID18's
[NAD83(2011) epoch 2010.0 / NAVD88 relationship](https://www.ngs.noaa.gov/GEOID/GEOID18/geoid18_tech_details.shtml).
The DFC files do not establish a WGS84 realization or coordinate epoch, so this
failure alone does not identify which input's datum interpretation explains
the difference. No alternative conversion was selected from the residuals.

2007: vertical datum **undocumented**, per the user's addendum that USGS has
only `No_metadata_supplied` for this project. Flight dates: March 8–17, 2007.
No absolute-height conversion was assigned to this survey. Its native metre
heights were compared directly with 2018 NAVD88 heights for relative stability.
Median 2007−2018 over common 2 m ground cells is **+0.05 m**; among stable
cells having ground in both surveys it is **+0.06 m** (20,263 cells).
These cell-wise results differ from the input's +0.12 m point-population median
comparison because the aggregations and sampling differ. No offset was removed.

## Raster and stability procedure

The 1,208 × 1,208 square grid uses 0.5 m pixels, aligns with the original DFC2019
pixel lattice, and encloses both ENU cells plus a 50 m margin. Extra padding
keeps the square metadata convention required by `read_truth`. Pixels beyond
the downloaded point box remain missing; there is no extrapolation.

DSM: maximum non-noise point, **all returns**, no interpolation. Classes 7 and
18 are excluded before rasterization. This matches the local
`dsm_validation.points_dsm` maximum-cell procedure. The original DFC lidar's
production rasterization rule cannot be established from local files, so exact
equivalence is not claimed. The class follows the maximum-height point.

2018 class mapping: 2→ground (2), 6→building (6), 5→vegetation (5), 9→water
(9), 17→bridge (17); class 1→code 5 as an explicitly **generic unclassified
non-ground proxy**, including tall structures, not a reliable vegetation
classification. Other/missing points→65. This prevents excluding class-1 roofs
while acknowledging that these inputs cannot give clean building/vegetation
classes. Per-class qualification uses **DFC2019 classes**; coverage uses these
mapped 2018 classes.

Stability uses maximum native heights in aligned **2 m cells**, with at least
**one non-noise point per survey**, and `abs(2018−2007) ≤ 0.5 m`. A qualifying
coarse cell supplies the mask to its 16 fine pixels. Missing fine DSM pixels
stay excluded. Every unstable fine pixel becomes CLS 65. No registration,
offset correction, gap filling or DFC-derived mask was used.

Files under `artifacts/reference/3dep-jax068-east/`:

- `2018/`: unmasked ellipsoidal DSM/CLS GeoTIFFs and `JAX_068_DSM.txt`.
- `stable/`: the rejected candidate key in the same validator-compatible format.
- `relative-stability.npz`: native 2007 DSMs at 0.5 m and 2 m, raw classes,
  2018 NAVD88 2 m DSM, point counts and 2 m stability mask. The 2007 data is
  deliberately kept outside the absolute truth directories.
- `qualification.json`: all measurements, fixed qualification limits and provenance.

## Qualification: fail

No registration. Pixels must be finite and labelled in both sources; water and
CLS 65 are excluded, consistently with `pipeline.dsm_validation`.

| Comparison | Pixels | MAE (m) | Median difference (m) |
| --- | ---: | ---: | ---: |
| Stable 2018, all labelled | 164,737 | **1.836377** | **−1.577118** |
| 2018 alone, all labelled | 247,439 | 2.046196 | −1.570142 |
| 2007 relative diagnostic only | 63,002 | 2.157655 | −1.531937 |

Fixed acceptance limits: all-labelled MAE **≤0.5 m** and absolute median
**≤0.3 m**. Both fail. The 2007 diagnostic compares its native heights with
DFC2019 expressed on the 2018 NAVD88 basis; it does **not** establish an
absolute datum for 2007 and does not participate in acceptance.

| DFC class | Stable 2018 MAE / median (m) | 2018 alone MAE / median (m) | 2007 relative MAE / median (m) |
| --- | ---: | ---: | ---: |
| Ground | 1.738 / −1.581 | 2.058 / −1.570 | 1.817 / −1.528 |
| Building | 1.911 / −1.574 | 2.034 / −1.574 | 2.285 / −1.530 |
| Vegetation | 1.758 / −1.545 | 2.038 / +1.040 | 4.058 / −3.144 |

There are no labelled water/bridge comparison pixels. Complete counts, bias,
RMSE and NMAD are in the JSON report.

## Stable coverage

Percentages below use the **entire cell/band area**, including missing pixels,
as denominator. Ground, non-ground proxy and building columns sum to all labelled.

| Region | All labelled | Ground | Code 5 proxy | Building |
| --- | ---: | ---: | ---: | ---: |
| Centre | 66.30% | 18.67% | 21.11% | 26.52% |
| East neighbour | 55.80% | 28.78% | 21.79% | 5.23% |
| ±10 m seam band | 57.78% | 28.68% | 9.21% | 19.89% |
| Two-cell union | 61.05% | 23.73% | 21.45% | 15.88% |

Among available labelled pixels, stability is respectively **66.44%**, **55.94%**
and **57.95%**. Per-class available-pixel denominators and the two-cell union
are also in the JSON. Water/bridge coverage is zero. Overall cell coverage
exceeds 20%, but several individual class subsets do not; adequate overall
coverage must not be used to pass an under-supported subset.

## Commands, tests and deferred work

```bash
export PYTHONPATH=
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PROJ_NETWORK=OFF
.venv-imagery/bin/python -m pipeline.lidar_reference --output artifacts/reference/3dep-jax068-east
.venv-imagery/bin/python -m pytest tests/imagery/test_lidar_reference.py tests/imagery/test_lidar_ept.py tests/imagery/test_dsm_validation.py -q
```

The builder exits **2** for this rejected candidate and refuses existing output
directories. The conversion was also repeated in a fresh temporary directory to
verify reproducibility; only reporting fields were added after the first run.
Tests include synthetic LAS and compressed LAZ, known geoid/frame anchor height,
preservation of EPT horizontal positions, maximum/non-noise rasterization,
half-open raster edges, missing data/counts, unremoved inter-survey offsets,
inclusive stability/qualification limits, empty qualification rejection,
missing geoid rejection, and validator truth-directory round trip.

Final verification: **20 tests passed** (11 new tests), `git diff --check`
passed. Expected rasterio warnings describe the existing DFC TIFFs' separate
text georeferencing and an upstream affine deprecation. The repeat build's
DSM/CLS TIFFs and metadata were byte-identical, and qualification was identical.

Changed/added repository files: `pipeline/lidar_reference.py`,
`tests/imagery/test_lidar_reference.py`, this document and local ignored
`tasks.md`. The eight new artifact files are listed above; source lidar,
metadata XML and original DFC files remain unchanged.

Steps 4–8 are deferred: no neighbour grid/configs, central boundary ownership
count or proposed measured re-tile, pre-registered neighbour decision file,
seam-check extensions, neighbour imagery/SfM/preflight statistics, training
profile or Stage B script. No model was graded against the rejected key.
The next action requires resolving the documented reference-frame/height
incompatibility; thresholds must not be loosened or a fitted DFC offset applied.

## Decision (2026-10-30): paused

The 3DEP key failed both pre-registered checks: absolute (`training/`-recorded qualification: −1.58 m median) and
shape (`training/3dep-shape-check.json`: ground MAE 0.333 m > 0.3 m after removing one uniform −1.577 m offset; other
rules passed). DFC2019 documents its heights as WGS84 Z, so the conversion was not changed. The true-neighbour
experiment is paused without loosening either rule.

Ready to resume: the `eastn` block is prepared through CPU preflight and `scripts/neighbour-stage-b.sh` refuses to
start until a qualified answer key, `configs/neighbour-decision.json` and the neighbour seam checker exist. Resume with
lidar closer to 2015, or with an answer-key test designed for multi-year lidar (e.g. robust agreement or masked
change), declared before it is computed and applied to a new area or reference, not re-grading this one.

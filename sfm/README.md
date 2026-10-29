# SatelliteSfM handoff for the Jacksonville pilot

Latest status (2026-10-29): the approved 2015 CORE3D pilot completed CPU sparse reconstruction, skew correction, Skyfall input conversion, measured Stage 1 training and holdout evaluation. The actual loader and GPU-enabled preflight pass; the trained run produced 102,543 splats with 214.9 MiB peak allocated tensor memory. See [the measured result](../training/RESULT_2015.md), [the training handoff](../training/PILOT.md) and [the revised plan](../docs/PLAN_V2.md). Fusion/export and Cesium 3D Tiles delivery are complete ([tiles](../training/RESULT_2015.md#3d-tiles-and-cesium)); independent geographic validation remains. Recent imagery selection remains deferred.

The approved pilot uses April 19–May 2, 2015: three training acquisitions and a
separate May 2 holdout. No four-view group satisfied the initial seven-day
policy. The user approved the shortest qualifying group, spanning 13.0035 days;
`configs/jax068-pilot.json` records its fixed acquisition IDs and the explicit
14-day selection ceiling.

## Reproduce preparation

```bash
bash scripts/prepare-jax068.sh
```

Requires Python 3.12+, uv and Git. The script creates a new timestamped run
folder, uses the isolated imagery environment, and fetches the pinned
SatelliteSfM source. It refuses a changed acquisition selection before image
download. The source cache is rehashed before reuse. This script was executed successfully
in a fresh run directory using the existing source cache. A new machine may download
about 5.98 GiB: eight PAN/MSI objects, bounded to 8 GiB total and 1 GiB each.
Images, crops, previews, camera files and local caches remain outside Git.

Individual stages are available as modules:

```bash
.venv-imagery/bin/python -m pipeline.acquisitions scan artifacts/imagery/catalogue.json --block configs/jax068-block.json --output artifacts/imagery/acquisitions.json
.venv-imagery/bin/python -m pipeline.acquisitions select artifacts/imagery/acquisitions.json --max-span-days 14 --output artifacts/imagery/selection-14d.json
.venv-imagery/bin/python -m pipeline.sources artifacts/imagery/selection-14d.json --output artifacts/imagery/selected-sources.json
.venv-imagery/bin/python -m pipeline.block_inputs artifacts/imagery/selection-14d.json artifacts/imagery/selected-sources.json --output artifacts/imagery/new-block
OPENBLAS_NUM_THREADS=2 .venv-imagery/bin/python -m pipeline.satellite_sfm artifacts/imagery/selection-14d.json --output artifacts/imagery/new-calibration
```

Each crop retains source pixels, metadata and the RPC model. The preparation
step verifies actual NITF RPCs against their selected sidecars, including crop
origins. Training and held-out images are written to separate directories.

## RGB preparation

MSI band roles are read from GDAL's explicit color interpretation (red 5, green
3, blue 2 in these WorldView-3 files), not guessed from a three-band slice.
PAN/MSI capture timestamps must match. MSI color is sampled through both RPC
models onto the native PAN sensor grid, using the pair's mean RPC height offset
as an explicit alignment reference. This is a reference coordinate, not a new
terrain-height measurement.

The alignment is tested across the configured height bracket. Preparation
rejects height-dependent displacement above 0.5 MSI pixels or less than 90%
valid color coverage. The four actual pairs have full valid coverage and maximum
height-dependent displacement of 0.0025–0.0254 MSI pixels.

Measured PAN intensity detail is injected after matching PAN mean/std to MSI
RGB intensity, with a ratio clipped to [0,4] and a shared 1–99 percentile byte
stretch. This produces relative display RGB, not calibrated radiance or a
validated color-accuracy product. It uses measured image detail, not generative
refinement. Pixel consistency with RPCs does not prove image-based registration
accuracy; inspect previews before feature matching.

The executed preview shows the same urban block in all four views, with no
obvious cloud obstruction. Vehicles, shadows and tint vary. This is a visual
screening observation, not independent quality or change-detection validation.

## Camera initialization and ENU

The adapter calls the actual pinned SatelliteSfM RPC approximation and
factorization code at `908c61937f6a16d6e1d50e98898db997e5e257bd`.
It fits a 15×15×7 sample grid and evaluates 2,048 distinct interior samples per
view. The observed maximum approximation errors are 0.062–0.101 pixels.
These quantify agreement with the supplied RPCs, not surveyed map accuracy.

The exact numerical ENU observer is preserved as
`enu_observer_latlonalt.json`: latitude 30.348819714393777°, longitude
-81.66376056134547°, ellipsoidal altitude -70 m. The altitude follows upstream's
lower-height-bracket-minus-20 m convention; it is not an estimate of ground
height. Both split input folders contain the same bounds and observer record.

The output retains full `K` and `W2C` matrices, including skew. The holdout gets
its own RPC-derived initial camera, but its pixels are excluded from training
SfM. These cameras must not be passed through an arbitrary new train/test split.
Upstream's default conversion uses a directory-order 90/10 split; the adapter
must preserve the approved acquisition roles instead.

## Sparse reconstruction and remaining gates

### Isolated CPU runtime

`scripts/setup-sfm-cpu.sh` builds ColmapForVisSat at
`79a6d150714d0eb350a1b10e17bedf91877568a1` with Ceres 2.1 at
`f68321e7de8929fbcdb95dd42877531e64f72f66`, installing only into
`.toolchains/sfm-cpu`. Python dependencies use `.venv-sfm`. This path avoids the
large container download and does not require GPU container passthrough.

The build uses existing Ubuntu development packages: a C++ compiler, CMake,
Boost, Eigen, FreeImage, glog/gflags, SQLite, GLEW/OpenGL and Qt5 development
headers. The script does not install system packages. CUDA, OpenGL execution,
CGAL and OpenMP are disabled. Explicit C/C++ standard-header includes and the
local Ceres include directory handle the old source's compiler/build-system
assumptions; source files remain unchanged.

```bash
bash scripts/setup-sfm-cpu.sh
.venv-sfm/bin/python -m pipeline.reconstruct --output artifacts/imagery/sfm-new-run
```

Run from the TerraSplat root and choose a new output directory for each attempt.
The runner checks source revisions, the recorded executable hash, matching
selection hashes and approved acquisition roles. It verifies source RGB hashes,
converts TIFF pixels to PNG without resampling, and copies full float64 cameras
including skew. The May 2 holdout remains in a separate directory, outside the
COLMAP image database.

SIFT extraction and matching use CPU mode. The reconstruction follows upstream's
32-pixel then 2-pixel triangulation/bundle-adjustment sequence, with its fixed
extrinsics, principal-point refinement and point-constraint weight 0.01.
Commands use checked subprocesses and individual logs. Failure or timeout leaves
the attempt and its report for inspection; no failed command is hidden by a
shell logging pipeline. Final reprojection errors are recomputed using the full
camera matrix after BA. This is internal consistency evidence, not independent
geographic accuracy validation.

### Executed result

The setup script completed on Ubuntu 24.04 with GCC 13.3 and CMake 4.3.2.
All six reconstruction commands passed in each of two fresh run directories.
The final verification run is `artifacts/imagery/sfm-2015-02`; its
`reconstruction-report.json` records command lines, timings, input/executable
hashes and final model hashes. [verified-result.json](verified-result.json)
retains the compact result and report checksum.

- All three training views registered; the database contains no holdout image.
- Final run: **2,233 sparse points**, 4,931 observations; 1,768 two-view tracks
  and 465 three-view tracks.
- Recomputed final reprojection error: median **0.274 px**, p95 **1.341 px**,
  maximum **2.056 px**. The 2 px triangulation filter precedes final BA and is
  not a strict bound on residuals after that BA.
- Principal-point changes are approximately −4.3 to +4.3 px. Fixed camera
  extrinsics and skew remain unchanged within numeric roundoff; the original
  ENU observer file is retained.
- The first run produced 2,221 points, median 0.270 px and p95 1.299 px.
  Multithreaded matching is not bit-identical; these runs establish successful
  execution, not identical model bytes or independent geographic accuracy.

Skew correction with a recorded image warp and explicit normalization have now
passed the [Skyfall handoff checks](../training/PILOT.md). Training and independent
placement/height validation remain pending.

### Container attempt history

This earlier container attempt did not execute. The isolated CPU path above
subsequently completed sparse reconstruction, resolving the runtime blocker.

The intended container is pinned to:

```text
ghcr.io/jayin92/satellitesfm@sha256:3c64997c9b2b4de5cdcad6d24972768423845fd7691015af1e4771c84fade9a5
```

At that attempt the local Docker runtime exposed `runc` but no NVIDIA runtime;
GPU container passthrough was unverified. The pinned image prefetch did not
complete, and a second attempt ended at its 180-second limit.
The image was not executed; the source-based RPC calibration above runs without
it. No system runtime
or GPU driver was changed by this preparation.

After SfM, preserve the ENU origin and every point/camera normalization, perform
skew correction with its pixel transform recorded, retain the explicit holdout,
and validate exported placement against independent evidence. Fitted camera
skew must not simply be dropped during Skyfall conversion.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-imagery/bin/python -m pytest tests/imagery -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-sfm/bin/python -m pytest tests/sfm -q
```

Tests cover crop geometry, metadata parsing, epoch gates, training-angle checks,
whole-view holdouts, input budget/cache integrity, RPC-aligned RGB, acquisition
matching and upstream camera metadata/observer conventions. The six SfM adapter
tests also check unchanged pixels, role isolation, subprocess failures,
skew-aware residuals and fixed-camera preservation.

Source reference: [SatelliteSfM preprocessing](https://github.com/jayin92/SatelliteSfM/tree/908c61937f6a16d6e1d50e98898db997e5e257bd).

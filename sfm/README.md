# SatelliteSfM handoff for the Jacksonville pilot

Planning update: the 2015 CORE3D inputs below are historical validation fixtures only. For a current-city model, qualify recent imagery and date-compatible validation before full training; see [the revised plan](../docs/PLAN_V2.md).

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

## Remaining reconstruction gates

Feature extraction, matching, bundle adjustment, sparse points, skew correction,
Skyfall normalization capture and training on this new block have not run.
Camera initialization is not a structure-from-motion result.

The intended container is pinned to:

```text
ghcr.io/jayin92/satellitesfm@sha256:3c64997c9b2b4de5cdcad6d24972768423845fd7691015af1e4771c84fade9a5
```

The local Docker runtime currently exposes `runc` but no NVIDIA runtime. Do not
assume GPU container passthrough works. A CPU SIFT execution path or configured
NVIDIA Container Toolkit must be verified before the SfM run. The pinned image
prefetch did not complete, and a second attempt ended at its 180-second limit.
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
```

Tests cover crop geometry, metadata parsing, epoch gates, training-angle checks,
whole-view holdouts, input budget/cache integrity, RPC-aligned RGB, acquisition
matching and upstream camera metadata/observer conventions.

Source reference: [SatelliteSfM preprocessing](https://github.com/jayin92/SatelliteSfM/tree/908c61937f6a16d6e1d50e98898db997e5e257bd).

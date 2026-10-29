# 2015 pilot: verified Skyfall inputs

The 2015 Jacksonville pilot has completed Stage 1, holdout evaluation and Cesium delivery. Five-view registered lidar MAE is 2.82 m; static MoGe depth supervision regressed it to 7.83 m and holdout PSNR from 17.73 to 15.18 dB. Retain the five-view baseline ([depth supervision](RESULT_2015.md#depth-supervision)); product web app v1 reading pipeline releases is next. This is one historical block with lidar of unknown date.

The current scene is `artifacts/imagery/skyfall-2015-02`.
[pilot-verified.json](pilot-verified.json) records the result and artifact hashes.
It is separate from the older, unregistered JAX_068 environment diagnostic in
[verified-result.json](verified-result.json).

## Reproduce conversion and checks

First prepare the [SfM runtime and sparse model](../sfm/README.md). The isolated
`.venv-sfm` requirements now include SciPy for resampling and Rasterio for source
validity masks. The registered-scene loader patch is applied to the isolated
Skyfall checkout by its existing setup script; the shared reference repository
is untouched.

From the TerraSplat root, using a new output directory:

```bash
python3 scripts/prepare-skyfall-stage1.py
.venv-sfm/bin/python -m pipeline.skyfall_inputs --output artifacts/imagery/skyfall-new-run
source scripts/stage1-env.sh
python scripts/check-skyfall-inputs.py --scene artifacts/imagery/skyfall-new-run
python -m pipeline.phase2 artifacts/imagery/skyfall-new-run/phase2-manifest.json --output artifacts/imagery/skyfall-new-run/preflight-report.json
```

The last command requires host GPU access. It checks the selected training
interpreter and three compiled extensions without starting training. The loader
check runs on a temporary copy because Skyfall writes a derived `points3D.ply`.
Generated scenes and logs remain ignored by Git. Regenerate the scene for its
intended workspace location; the preflight manifest records the local backend
and training interpreter.

## Image and camera conversion

For each camera, the pixel transform is
`H = [[1, -skew/fy, offset_x], [0, 1, offset_y], [0, 0, 1]]` and
`K_corrected = H × K`. The canvas contains the transformed source pixel-center
corners. It is expanded rather than cropped back to the original width.
Each view records `H`, its inverse, both image sizes and the resulting camera.

RGB uses bilinear resampling in integer pixel-center coordinates. Validity comes
from the original GeoTIFF dataset mask, checked against preparation evidence;
source TIFF and SfM PNG pixels must agree. Interpolation footprints containing
invalid data and padding are masked out. Valid black pixels remain valid.

The three training views use bundle-adjusted intrinsics and fixed ENU poses.
The May 2 holdout uses its original RPC-derived camera without a fitted
correction. The approved three-train/one-test split is explicit; no directory
order or 90/10 re-split is used. Dimensions and intrinsics are stored per frame.

## Normalization and placement

Only the 2,233 training sparse points determine normalization. ENU is already
Z-up, so the additional rotation is identity. The center is the coordinate-wise
median, scale is `256 / percentile99(distance_to_center)`, and the first
percentile of scaled Z is subtracted:

`model = scale × (ENU − center) − [0, 0, z_shift]`.

The executed scale is approximately 1.174573579 model units/metre, with center
`[16.3178751312, 11.8861251988, 49.3221451964]` metres and Z shift
`−8.0728847063` model units. The original ENU observer remains
`[30.3488197144°, −81.6637605613°, −70 m]`; its altitude is a numerical origin,
not a measured ground height.

`georeference.json` binds the prepared files to the captured normalization and
stores row-major ENU↔model and model→ECEF matrices using the existing placement
implementation. Camera rotations remain rigid; scale applies to camera centers
and points. The loader receives already normalized transforms without legacy
`R`/`T` fields. Its registered-scene branch does not normalize again or fall back
to random initialization points.

The scene contract requires `--eval`, checks payload hashes and acquisition IDs,
and rejects a second normalization. The preflight profile
`basic-stage1-registered` accepts only the exact recorded Stage 1 and prepared
ENU patches; arbitrary source edits remain rejected. Legacy inputs retain their
existing loader behavior.

## Executed checks and next step

- Four corrected views loaded: three training cameras and one holdout.
- Original source masks are fully valid; deskewed canvas validity ranges from
  91.45% to 99.69%, with padding excluded.
- Maximum deskew/normalized-camera projection difference: **1.126 × 10⁻⁹ px**.
- Maximum ENU normalization round-trip difference: **5.685 × 10⁻¹⁴ m**.
- Actual loader camera projection difference: at most **1.526 × 10⁻¹⁰ px**.
- Float32 PLY point encoding difference: at most **1.324 × 10⁻⁵ model units**.
- Missing `--eval`, changed input checksums and a second normalization were
  rejected by the actual loader. CUDA nearest-neighbor, SSIM and rasterizer
  forward/backward checks passed, as did GPU-enabled scene preflight.

These are numerical consistency and execution checks. They do not establish
surveyed placement or training quality on this scene.

Measured Stage 1 training and evaluation have now completed on a fresh copy of
the prepared inputs. The 30,000-iteration run produced 102,543 splats with
214.9 MiB peak allocated tensor memory; the held-out May 2 view scored 17.57 dB
PSNR and training views 30.05–42.37 dB. Those are masked appearance metrics, not
height or placement accuracy. See [the result](RESULT_2015.md) for the full
table, hashes and limitations. The basic Stage 1 restrictions still apply: IDU
and pseudo-depth modes are disabled. Fusion/export preserving the recorded placement
is complete ([fused export](RESULT_2015.md#fused-export)), and so is [3D Tiles delivery in CesiumJS](RESULT_2015.md#3d-tiles-and-cesium); an [independent lidar check](RESULT_2015.md#independent-lidar-check) measured a 1.8 m horizontal offset, within about 0.2–0.6 m vertically, and [five training views](RESULT_2015.md#five-training-views) cut height MAE from 3.3 m to 2.8 m.

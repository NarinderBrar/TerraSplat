# Skyfall Stage 1 tiles

Skyfall-GS Stage 1 trained by TerraSplat on the authors' prepared scene;
visually registered by TerraScope (about 1.3–3.8 m); vertical approximate;
not produced from TerraSplat's raw-imagery pipeline. These are demonstration
models, not the authors' finished models or production reconstructions.

## Inputs and recipe

Prepared data: `jayinnn/Skyfall-GS-datasets`, revision
`82bcd8ae044afcf3182d335a47d0694ba2b4daa2`, 2,337,656,814 bytes at
`../Ref/Skyfall-GS/data`. The local download manifest records every file hash.
The pinned [dataset card](https://huggingface.co/datasets/jayinnn/Skyfall-GS-datasets/blob/82bcd8ae044afcf3182d335a47d0694ba2b4daa2/README.md) declares Apache-2.0.
No finished `*_final.ply` is used. Reference inputs and TerraScope stay read-only.

`configs/train-ref-jax068-recipe.json` fixes the authors' Stage 1 flags:
30k iterations, appearance model, SH 1, live MoGe pseudo-depth weight 0.5,
opacity weight 10, no static depth loss. Resolution 2 and a 200k doubling-headroom
splat cap fit the 6 GB GPU. Holdouts remain excluded from training, priors and
choices. JAX_068 reuses its existing completed run; the other scenes run serially.

| Scene | Train / test views | Splats | Minutes | Peak GiB | Holdout PSNR | Tiles / levels | GLB bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| JAX_068 | 17 / 2 | 149,842 | 53.4 | 2.90 | 18.90 | 17 / 3 | 2,123,248 |

## Placement and crop

`pipeline.skyfall_scenes` converts the registration raster affine and z18 tile
origin to the geographic model origin, removes Mercator scale with cos(latitude),
and produces uniform scale, rotation about Up and translation in true ENU metres.
The rigid ENU-to-ECEF root transform is applied once. Tiling rotates both centres
and covariance quaternions. Each prepared sparse core is checked against the
registration raster; legacy normalization is identity for all twelve inputs.
This verifies frame compatibility, not Stage 1 landmark accuracy.

TerraScope's calibrated street `groundZ` maps to approximate city street altitude
(12 m orthometric Jacksonville, 5 m New York), converted with local EGM96 via
pyproj when installed. Without that grid, documented city undulations are
−29.4 / −32.7 m. These heights are approximate and are never fitted to lidar.
Derived placement manifests retain the original export hash; training data and
original exports stay immutable. Existing rendered-depth points are transformed
through model space for JAX_068; each release records their hash and modal ground.

`configs/tiles-skyfall.json` fixes a 500 m ENU core centred on the visual origin,
excluding border context and floaters. NYC_336 uses a 350 m core centred at
model (115, 0), contained within TerraScope's conservative land footprint
(model X −60 to 290); river water is excluded. Removed counts are retained in each plan.
JAX_068 retains 77,021 / 149,842 centres, removing 72,821. The pilot profile and
its export gates remain unchanged. Leaves preserve the full retained model;
parents use moment-matched merges and REPLACE refinement. Every bound includes
3σ support and all descendants; geometric errors strictly decrease. SH colour
is baked to DC over training directions only. SPZ v2 is carried in glTF
`KHR_gaussian_splatting` inside 3D Tiles 1.1. JAX_068 maximum SPZ errors are
0.122 mm position, 0.001961 activated opacity and 0.031251 log scale. No log-scale
components lie outside the SPZ range [−10, 5.9375], so none are clamped.

`3d-tiles-validator` reports no tileset structure issues for JAX_068; its 17
content errors come from unsupported Gaussian extensions and their attribute
names, matching the pilot limitation. Actual decoding and rendering pass.

## JAX_068 lidar check

The visual registration places the model 1.80 m horizontally from DFC2019 lidar
(−1.0 m east, +1.5 m north); vertical offset is +4.99 m. Registered rendered-depth
MAE is 1.38 m. This checks TerraScope's visual placement separately from model
shape; it is not surveyed control. No lidar-derived correction is applied.
The rendered-depth ground mode is −17.46 m ellipsoidal, versus −20.75 m for the overlapping
pipeline block (3.29 m apart). The overlap switch selects the corresponding
terrain patch; heights are not averaged between hidden models. Compact evidence: `training/skyfall-jax-068.json`.

## Status

JAX_068 is published locally as `skyfall-jax-068-stage1-r1`. The app uses per-dataset tilesets, rather than a regional index with opaque children,
so layer and overlap controls can hide each dataset before content requests begin.
The default overlap is Skyfall; other datasets stream independently. Cache totals
are budgeted at 256 MiB plus 25% overflow across datasets, maximum screen-space
error 12, request culling while moving enabled. Diagnostics show loaded tiles and
memory bytes. Hover uses geometric hit tests without per-move pick rendering.
Terrain uses the nearest visible dataset's rendered ground patch.

Offline browser checks load zero GLBs at the globe, only the JAX_068 root at 8 km,
and 13 of 17 tiles through a fixed close camera. Layer and overlap switching,
pipeline release substitution, splat coverage and zero page/external-request errors
pass. JAX, pipeline and globe screenshots were inspected. Remaining scenes and
NYC validation are pending. The fixed batch configuration is
`configs/skyfall-scenes.json`; source `scripts/stage1-env.sh` and run
`.venv-imagery/bin/python -u -m pipeline.skyfall_batch`. Scene status and logs live
under `artifacts/skyfall/`. Completed stages are reused; failures preserve their
runs and are skipped while subsequent scenes continue. Disk free space is checked
before each scene; GPU work stops below 5 GiB. Never remove pre-existing runs.

Stage 1 details remain softer than the authors' final multi-million-splat models.
Visual registrations were measured on those final models; shared prepared frames
are compatible, but local Stage 1 distortions can differ. Approximate flat ground
patches may have seams. Capture dates are unavailable in the prepared inputs;
NYC scenes are synthetic. No production or raw-imagery provenance is claimed.

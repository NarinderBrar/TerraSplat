# 2015 pilot: measured Stage 1 result

The fresh run `artifacts/stage1/pilot-2015-04` completed 30,000 iterations and
passed final checkpoint/PLY validation. The May 2 view was held out from
training and model selection. This is a historical demonstration on one small
block; it is not independent height or placement accuracy evidence.

Compact checked-in evidence: [result-2015-04.json](result-2015-04.json).
Models, checkpoints, renders and logs remain in ignored `artifacts/`.

## Result

- Status: `trained`; both phases returned 0 and the 1,000-iteration checkpoint
  resumed with the full 30,000-step schedule.
- Splats: **102,543** at iteration 30,000, produced from 2,233 sparse points.
  The 200,000 cap was not reached; 153 densification steps were skipped by the
  doubling-headroom rule.
- Wall time: **541 s** (about 9 minutes) for both phases.
- Peak PyTorch allocated tensor memory: **225,390,592 bytes (214.9 MiB)**.
  Peak reserved: 358,612,992 bytes (342.0 MiB). Maximum sampled total device
  usage, including driver/context and any other process, was 564,396,032 bytes
  (538.2 MiB), well under the 4.5 GiB budget.

## Held-out appearance metrics

MSE and MAE are over RGB samples at valid mask pixels; PSNR uses unit-range
colors and a 1e-12 MSE floor. Alpha coverage is the fraction of valid pixels
with rendered alpha above 0.01.

| Role | Catalog | MSE | MAE | PSNR (dB) | Alpha coverage | Mean alpha |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train | 104001000AB24A00 | 0.000058 | 0.002871 | 42.37 | 1.0000 | 0.799 |
| train | 104001000A1B5F00 | 0.000706 | 0.008659 | 31.51 | 1.0000 | 0.793 |
| train | 104001000AB10600 | 0.000989 | 0.009329 | 30.05 | 1.0000 | 0.732 |
| holdout | 104001000A22A000 | 0.017487 | 0.084566 | 17.57 | 0.9822 | 0.768 |

The initial sparse model scored 14.76–15.39 dB PSNR on the four views. The
final train-view error is low; the holdout error is much larger. The holdout
camera is the uncorrected RPC-derived camera, so that gap includes cross-date
appearance change and camera/registration difference, not only model quality.

Visual inspection of `evaluation/comparison.png`: the final holdout render lines
up with the reference (building outlines, roads and parking lots fall in place),
but it takes on the warmer colour balance of the April 19 training view
(`104001000AB24A00`), not the holdout's green-tinted balance. Without an
appearance model, the cross-date colour mismatch therefore accounts for much of
the holdout error. Floaters and smears appear near the image borders: the
right edge of the holdout, the left edge of `104001000AB10600` and the
bottom-right corner of `104001000A1B5F00`. The initial sparse renders are a
nearly uniform green field.

## How the result was produced

- Profile `jax068-2015-stage1-baseline`: 30,000 iterations, resolution factor 2,
  SH degree 2, densification from 1,000 to 20,000, no appearance MLP, depth
  supervision, pseudo-depth or generative refinement.
- Three training acquisitions; `104001000A22A000` (May 2, 2015) is the whole-view
  holdout. No holdout render or image loss was used during training or profile
  selection.
- Spherical harmonics use Skyfall's existing PyTorch path
  (`convert_sh_python: true`). The CUDA SH path produced nonfinite renders on
  this scene; only the renderer option changed, not geometry or data.
- Measured runs clamp the raw opacity logits to `[-13.8, 13.8]` after
  `reset_opacity`, because upstream inverse-sigmoid at an exact probability
  endpoint can store `+inf`. A NaN check runs after the reset and opacity
  finiteness is checked every 100 iterations.

## Run history and limitations

- `pilot-2015-02` reached 30,000 iterations but failed final validation with a
  nonfinite checkpoint tensor (an opacity logit at exactly 1.0 after activation).
  Its report remains `failed`.
- `pilot-2015-03` was interrupted with the previous working session at
  iteration 8,300 of 30,000 and was not resumed. Its local report now says
  `interrupted`; it is not a successful run.
- `pilot-2015-04` is the completed run above.
- The evaluator needed a local PLY loader because Skyfall's `load_ply` only
  restores `filter_3D`; its geometry attributes are expected to come from a
  checkpoint. The evaluator now reads the raw PLY attributes directly.
- Tiling, Cesium delivery and independent geographic validation remain future
  work; `georeference.json` accuracy is unverified.

## Fused export

`python -m pipeline.export --run artifacts/stage1/pilot-2015-04` verifies the
training hashes and runs Skyfall's `create_fused_ply.py` from the checkpoint. It
then validates the result and records the placement contract the tiler will
use. Compact evidence: [export-2015-04.json](export-2015-04.json).

- Fused PLY: 102,543 splats, the same as the trained model, SH degree 2.
  `sha256 b2d5b9a5…dd7a`, 16.8 MB. The 3D filter is folded into opacity and
  scale, as Skyfall's `save_fused_ply` does; no appearance colour mapping
  applies.
- All attributes are finite. Activated opacity is strictly inside (0, 1), with
  a maximum of 0.99958, and all scales are positive.
- Transform chain: model → ENU → ECEF → ENU → model round-trip error is
  1.4e-9 m. `model_to_ecef` matches `enu_to_ecef · model_to_enu` to 1.0e-9 m.
  The gate is 0.0335 m, which is 0.1 × the finest 0.335 m native GSD. This
  proves only that the recorded transforms are consistent with each other; it
  is not a placement accuracy measurement.
- Render equivalence: the fused PLY and the raw final model, rendered from the
  same cameras, differ by 105.9–114.0 dB PSNR across all four views, which is
  numerical noise.
- Extent: splat centres span 557 × 472 × 179 m in ENU. Only 38.5% lie within
  the nominal 145 m half-extent (250 m block plus 20 m halo); the
  opacity-weighted p95 radius is 224 m. Floaters at the image borders spread
  beyond the block, so the tiler must crop or flag them.
- The manifest records `model_to_ecef` (column-major) as one valid root
  transform. The tiler below uses the equivalent rigid form instead.
- Tiler rules: `REPLACE` refinement only. Child geometric error is non-zero
  and strictly below the parent's. Every tile and all its ancestors must
  contain each splat's centre plus about 3σ of its extent. Placement uses
  `root.transform`, not `CESIUM_RTC`.

## 3D Tiles and Cesium

```
python -m pipeline.tiling --export artifacts/stage1/pilot-2015-04/export --output artifacts/tiles/pilot-2015-04
node scripts/encode-tiles.mjs artifacts/tiles/pilot-2015-04
npm run build && npx playwright test tests/pilot.browser.js
```

Run the planner with `.venv-stage1/bin/python`, which has `plyfile`. The
generated tiles stay in `artifacts/tiles/`; `public/tiles/` links them for
`pilot.html`. Compact evidence: [tiles-2015-04.json](tiles-2015-04.json).

- Placement: `model_to_enu` (uniform scale 0.8514 plus translation, no
  rotation) is baked into the splats, so tile content is metric ENU and
  `root.transform = enu_to_ecef` stays rigid. This is mathematically the same
  as `model_to_ecef`, but it avoids relying on Cesium to scale splat
  covariances through a non-rigid root transform. Payloads use glTF Y-up axes
  (east, up, −north); a unit test checks that positions and covariances
  rotate exactly.
- Crop: an ENU square of ±125 m around the anchor keeps 36,722 of 102,543
  splats. The rest are outside the core block, either on ground the training
  crops covered beyond the block or as border floaters.
- Tree: 17 tiles in 3 levels, an adaptive quadtree with at most 8,192 splats
  per tile. Leaves hold the full-resolution splats; parents hold
  moment-matched voxel merges. Geometric error is 13.5 m at the root, 3.4 or
  1.7 m at merged quadrants and 0.335 m (1 GSD) at the leaves. The tileset
  error is the root box diagonal (696 m), the error of drawing nothing.
- Encoding: SPZ v2 inside `KHR_gaussian_splatting` glb files, the Phase 0
  profile, now shared through `scripts/lib/splat-glb.mjs`. The fixture output
  is unchanged. 64,511 splats across all levels take 1.16 MB. Maximum
  quantization error is 0.12 mm in position, 0.002 in opacity and 0.031 in
  log scale. Decoded centres stay inside their tile boxes.
- Colour: SH degree 0. The view-dependent SH is folded into the base colour
  as the mean colour over the three training directions. Higher-degree SH
  awaits a verified Cesium SH convention.
- Validation: `validate_tree` enforces the tiler rules. `3d-tiles-validator`
  0.6.1 reports no tileset-structure issues. Its 17 errors are all content
  errors: it does not support `KHR_gaussian_splatting` and rejects that
  extension's attribute names, which the Phase 0 fixture also uses.
- Cesium 1.146.0, headless browser with SwiftShader: no errors and no
  external requests. From the top view, 91.5% of rendered pixels fall inside
  the projected 250 m block square and 99.1% of the square is covered, with
  north up and buildings matching the source imagery. From 8 km only the
  merged root draws; close up, it is replaced by full-resolution leaves.

Limitations:
- The footprint check compares the render with the recorded transform chain;
  it is not independent placement or height accuracy.
- Oblique views show needle-shaped splats along the satellite view rays,
  because three views constrain height poorly. This is not a tiler defect.
- About 10k of the owned splats reach more than 10 m (3σ), so tile bounds are
  loose.
- The crop is an ENU square, while the source block is a UTM square rotated
  by about 0.3°. Multi-block seams will need a shared ownership grid.
- The ground lies about 17 m below the ellipsoid and no terrain is loaded,
  so `pilot.html` hides the globe, or makes it translucent for the optional
  OpenStreetMap overlay.

## Independent lidar check

```
source scripts/stage1-env.sh
python -m pipeline.render_depth --run artifacts/stage1/pilot-2015-04 \
  --output artifacts/validation/pilot-2015-04-depth
.venv-imagery/bin/python -m pipeline.dsm_validation --export artifacts/stage1/pilot-2015-04/export \
  --truth artifacts/reference/dfc2019-jax068 --output artifacts/validation/pilot-2015-04-depth/lidar \
  --points artifacts/validation/pilot-2015-04-depth/depth-points.npz
```

Without `--points`, `dsm_validation` uses the cruder splat-footprint DSM.

Reference: the DFC2019 Track 3 truth for JAX_068, an HSIP airborne lidar DSM at
0.5 m in UTM 17N with ground, building and vegetation classes. It covers the
whole block; its centre is within 1.4 m of ours. Heights are WGS84 ellipsoidal:
the ground median is −17.4 m, where orthometric heights would be about +12 m
here. It is independent of our imagery, RPCs and SfM. The three files were
range-extracted from the Sat-NeRF `EarthVision2022` `dataset.zip` into ignored
`artifacts/reference/`; their hashes are in
[validation-2015-04.json](validation-2015-04.json). The lidar flight date is
not published.

Both DSMs are placed only through the recorded transform chain: model → ENU →
ECEF → UTM with ellipsoidal heights. Skyfall-GS's `dsmr` then registers each
DSM to the lidar; a synthetic test fixes the sign of its shift.

- Rendered depth (primary): the fused model is rendered from a synthetic,
  nearly orthographic nadir camera: 17 km altitude, 0.98° field of view,
  0.25 m pixels. Building lean is at most 0.26 m for 30 m at the block edge.
  Expected depth (alpha-weighted depth divided by alpha, alpha ≥ 0.5) is
  back-projected, and the maximum height per 0.5 m cell is kept.
- Splat footprints (diagnostic): each splat with opacity ≥ 0.5 is drawn as a
  disc at its centre height.

| 250 m block, rendered depth | Bias | Median | MAE | NMAD | Completeness |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unregistered, all labelled | +0.15 m | −0.19 m | 3.50 m | 3.16 m | 99% |
| Unregistered, ground | +0.62 m | +0.16 m | 2.99 m | 2.92 m | 99% |
| Unregistered, buildings | −0.20 m | −0.49 m | 4.01 m | 3.36 m | 99% |
| Registered, all labelled | 0.00 m | −0.40 m | 3.32 m | 2.99 m | 99% |
| Registered, ground | +0.28 m | −0.13 m | 2.93 m | 2.85 m | 99% |
| Registered, buildings | −0.16 m | −0.63 m | 3.70 m | 3.08 m | 99% |

The footprint DSM reads −1.07 m bias, 4.54 m MAE and 82% completeness
unregistered.

- Horizontal placement: both DSMs give the same offset. The splat surface
  sits 1.5 m west and 1.0 m south of the lidar, 1.8 m horizontally or about
  5 GSD, resolved on the 0.5 m grid. No ground control was used; this is
  within the few-metre accuracy typical of WorldView-3 RPCs without it.
- Vertical placement: with rendered depth the surface is within about
  0.2–0.6 m of the lidar (mean +0.15 m, median −0.19 m; dsmr offset
  +0.24 m). The footprint DSM's −1.07 m came from using splat centres.
- Shape: MAE is about 3 m on ground and 3.7 m on buildings after
  registration. Edges and roofs are smeared because three satellite views
  constrain heights poorly; the comparison image shows building-sized
  over- and under-estimates. These values come from one nadir camera and a
  Stage 1 model trained on 3 views, so they are not directly comparable to
  Skyfall's published numbers.
- The lidar date is unknown, so some disagreement may be scene change.

## Five training views

Two more acquisitions were added to test whether more viewing angles improve
heights. [jax068-pilot-8w.json](../configs/jax068-pilot-8w.json) widens the
window to April 19–June 15, 2015 (57 days) and adds May 21 (5.9° off nadir)
and June 15 (11.1°). May 14 fails the 10% cloud gate. The May 2 holdout and
all Stage 1 settings
([train-2015-pilot-8w.json](../configs/train-2015-pilot-8w.json)) are
unchanged, so the run compares directly with `pilot-2015-04`. Selection uses
`acquisitions select --approved`, which applies the same coverage, cloud,
time-window and angle gates to a named view set. The six RGB crops were
reviewed by eye: same cloud-free block, varying lighting and vehicles.
Compact evidence is in [result-2015-8w.json](result-2015-8w.json).

| | 3 views (`pilot-2015-04`) | 5 views (`pilot-2015-8w-01`) |
| --- | ---: | ---: |
| Sparse points (median / p95 residual) | 2,233 (0.27 / 1.34 px) | 5,618 (0.31 / 1.37 px) |
| Splats | 102,543 | 107,464 |
| Holdout May 2 PSNR | 17.57 dB | 17.73 dB |
| Training-view PSNR | 30.1–42.4 dB | 24.0–38.1 dB |
| Lidar, registered MAE, all labelled | 3.32 m | **2.82 m** |
| ground / buildings / vegetation | 2.93 / 3.70 / 3.75 m | 2.77 / 2.87 / 2.78 m |
| Registered RMSE / NMAD | 5.04 / 2.99 m | 4.17 / 2.82 m |
| Horizontal offset from lidar | 1.8 m (1.5 W, 1.0 S) | 1.6 m (1.5 W, 0.5 N) |
| Unregistered median height error | −0.19 m | −0.92 m |
| Completeness (rendered depth) | 99% | 99.6% |

- Heights improved most on buildings (−22% MAE) and vegetation; ground
  barely changed. Building outlines are visibly sharper in the DSM, though
  roof errors of several metres remain.
- Holdout PSNR barely moved. Training-view PSNR fell because five dates
  without an appearance model cannot each be fitted exactly; that is less
  overfitting, not a regression.
- The surface now sits about 0.8–0.9 m low before registration (dsmr offset
  −0.82 m), versus within 0.2–0.6 m before. Absolute height still rests on
  RPCs without ground control.
- The fused export round trip is 1.5e-9 m and the fused render matches the
  trained model (≥101 dB).

The Cesium pilot now shows `pilot-2015-8w-01`; compact tile evidence is in
[tiles-2015-8w.json](tiles-2015-8w.json). With the same tiling profile, 3 → 5
views gives 17 → 21 tiles (both 3 levels), 36,722 → 61,857 owned splats and
1.16 → 1.54 MB of glb content. Top-view inside fraction is 91.54% → 91.33%
and block coverage is 99.07% → 98.56%. The browser test passes unchanged,
including distance refinement, with no errors or external requests. Both
screenshots were reviewed; oblique needle-like splats and border streaks
remain. Large splats hit the SPZ v2 scale ceiling: maximum log-scale error
is 1.197 versus 0.031 for three views (position error stays 0.12 mm). The
footprint check tests the transform chain only, not independent placement
or height accuracy.

## Depth supervision

`pilot-2015-8w-depth-01` repeats the five-view run with static MoGe camera-z
priors and fixed `lambda_depth: 0.5` (Skyfall default). The May 2 holdout is
excluded from priors, training and setting choices; neither holdout nor lidar
was used to tune the weight. IDU and pseudo-depth remain disabled. Compact
evidence: [result-2015-8w-depth.json](result-2015-8w-depth.json).

MoGe commit `aca893a40faf6905cb722d6614cc37cb87ea06f7` runs separately in
`.venv-moge` with CUDA PyTorch 2.7.1; exact versions and source setup are in
[requirements-moge.txt](requirements-moge.txt). `Ruicheng/moge-vitl` weights
are pinned to HF revision `ad326bfb61facd6c52b5a825bc1e34d7c97d9672`.
`pipeline.depth_priors` reads full-resolution training RGBs and supplies each
camera's horizontal FOV (0.012–0.060°) to `infer()`. Default inference level 9
returns full-sized depth maps; invalid mask/nonfinite pixels become NaN in
single-channel float32 EXRs. Valid fractions are 91.45–99.81%. The new
`skyfall-moge` scene hashes the priors and preserves the georeference. The
actual loader and GPU preflight pass; Skyfall sources are unchanged.

Skyfall resizes depth by nearest neighbour at training resolution 2, masks it,
zeroes nonfinite samples and applies `1 − Pearson correlation`. Logged EMA
depth loss is non-zero (0.000057–0.772; final 0.000383). Training completed
30,000 iterations in 669 s with 100,476 splats and 214.7 MiB peak allocated
memory. Fused export passed (1.4e-9 m round trip, ≥96.1 dB render equivalence).

| | 5 views | 5 views + MoGe depth |
| --- | ---: | ---: |
| Holdout May 2 PSNR | 17.73 dB | 15.18 dB |
| Training-view PSNR | 23.98–38.08 dB | 16.78–17.21 dB |
| Lidar, registered MAE, all labelled | 2.82 m | 7.83 m |
| ground / buildings / vegetation | 2.77 / 2.87 / 2.78 m | 7.48 / 8.51 / 3.94 m |
| Registered RMSE / NMAD | 4.17 / 2.82 m | 10.66 / 5.94 m |
| Horizontal offset returned by registration | 1.6 m (1.5 W, 0.5 N) | 38.9 m (37.5 E, 10.5 S) |
| Unregistered median height error | −0.92 m | +16.80 m |
| Registered completeness | 99.6% | 84.1% |

The depth-supervised DSM is nearly flat and loses roof structure. Its 38.9 m
registration shift is unreliable; registered errors use only 84.1% overlap.
Unregistered MAE is 15.51 m.

Diagnosis: static priors are around 1,900–7,800 units with only about 0.2 units
of variation within each image. Skyfall sets masked/padded pixels to zero
before Pearson correlation. The inside-versus-padding step dominates the
loss and mainly enforces flat depth. This run tested a padding-dominated
flattening loss, not MoGe's shape signal. Valid-pixel correlations against
the five-view baseline's expected depth are 0.746, 0.965, −0.129, 0.483 and
0.175 for AB24A00, A1B5F00, AB10600, C22D200 and C478800, respectively
(full IDs in the compact JSON).

The authors disable static satellite-image priors (`--lambda_depth 0`) and
instead use MoGe on rendered pseudo views inside training
(`--lambda_pseudo_depth 0.5`), an appearance model and `--lambda_opacity 10`.
Their pseudo-view variant was untested by this experiment. Retain the
five-view baseline; no weight tuning used holdout or lidar. One block,
lidar date unknown; no broader accuracy claim.

Checks for the static-prior experiment: 63 SfM/Python tests and 34 imagery
tests pass; Ruff E/F passes.

## Authors' JAX_068 data

The authors' prepared scene has 17 training views, two test views and 69,027
initial points. `pipeline.reference_scene` copies it without changing `Ref/`.
The legacy loader skips normalization because this scene has no `R`/`T`:
centre `[0, 0, 0]`, scale 1, z-shift 0. The real-loader comparison found
zero point/camera difference. The missing ENU anchor is **assumed** from the
lidar AOI bounds centre, with altitude −70 m (pilot lower bracket minus 20 m).
Only registered shape metrics are meaningful here.

Both profiles were fixed before evaluation. The first uses the five-view
pilot settings. The second uses the authors' README Stage 1 recipe: live
MoGe pseudo views, appearance model, opacity weight 10 and SH 1; resolution 2
and the 200k doubling-headroom cap adapt it to this GPU. Static depth is off.
The original basic adapter is byte-identical and the pilot preflight passes.
Compact evidence: [result-ref-jax068.json](result-ref-jax068.json).

| Rendered-depth DSM, registered | Holdout PSNR (dB) | MAE all (m) | Ground | Buildings | Vegetation | RMSE (m) | NMAD (m) | Completeness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pipeline, 3 views | 17.57 | 3.32 | 2.93 | 3.70 | 3.75 | 5.04 | 2.99 | 99.0% |
| Pipeline, 5 views | 17.73 | 2.82 | 2.77 | 2.87 | 2.78 | 4.17 | 2.82 | 99.6% |
| Reference, 17 views, pilot settings | 21.35 | 2.36 | 2.67 | 2.06 | 1.93 | 4.03 | 1.87 | 99.9% |
| Reference, 17 views, authors' recipe | 18.90 | **1.38** | 1.24 | 1.52 | 1.62 | 2.43 | 1.01 | 100.0% |

The reference-data run lowers MAE by 0.46 m (16%) from five views, with the
largest gain on buildings. This includes different cameras, images and sparse
initialization, so it cannot be attributed to view count alone. The reference
PSNR is the mean of its two test-frame PSNRs (22.44 and 20.27 dB); those are
different holdouts from the pilot's May 2 frame.

For the pilot-settings run, dsmr returns `(dx, dy) = (0, −2)` pixels and a
`+24.68 m` z correction. The search covers ±300 pixels per axis at full grid
resolution (±150 m); the solution is well inside it, with 99.9% overlap.
These shifts are **not placement measurements**. The full-cloud extent is a
diagnostic for the wider reference scene; the pilot extent gate stays enforced.
The exported fused/raw renders agree to at least 103.4 dB.

The recipe memory trial crossed pseudo-depth start at step 1,000 and completed
1,500 steps: 2.53 GiB peak allocated, 3.22 GiB reserved. Pinned MoGe and cached
weights import in `.venv-stage1` with torch unchanged. A warmup checkpoint
validation import error was fixed, its evidence retained, and the full run
restarted from scratch. The full recipe completed 30k steps in 53.4 minutes
with 149,842 splats and 2.86 GiB peak allocated memory (3.43 GiB reserved).
Its fused/raw renders agree to at least 113.9 dB with the authors' fixed
appearance embedding; no holdout appearance fitting is performed.

On the same 17 views, the joint recipe cuts another 0.97 m (41%) from MAE,
with ground 2.67 → 1.24 m and buildings 2.06 → 1.52 m. This is a recipe
comparison, not an isolated MoGe ablation. Holdout PSNR falls by 2.45 dB:
the render is smoother and changes colour, despite better DSM shape.
The recipe's registration is `(0, −1)` pixels and `+24.68 m` z correction,
with 100% overlap; it also lies well inside the search and is not placement
evidence. Widen the pipeline view set and test this recipe as a geometry
profile, while retaining the known-anchor five-view viewer baseline.

Checks: 68 SfM/Python tests pass (two pyproj-dependent tests skip there and
pass in the imagery environment), 34 imagery tests pass, and Ruff E/F passes.
All four reference-adapter tests pass in the imagery environment.

One block, authors' cameras rather than pipeline cameras, assumed anchor and
lidar date unknown; no placement-accuracy or production-readiness claim.

## Authors' recipe on the pilot

`pilot-2015-8w-recipe-01` applies the same authors' recipe to the unchanged
five-view scene and its captured ENU anchor. The
[profile](../configs/train-2015-pilot-8w-recipe.json) was fixed before evaluation:
live MoGe pseudo views, appearance model, opacity weight 10, SH 1, resolution 2
and the 200k doubling-headroom cap; static satellite depth stays off. The
runner selects and preflights the recipe adapter through a separate manifest,
without changing the scene. All 30 source files remain unchanged. Compact
evidence: [result-2015-8w-recipe.json](result-2015-8w-recipe.json).

Pseudo cameras are synthetic 60° orbit views around random model-space targets.
Their appearance IDs sample only `0..len(scene.getTrainCameras())-1` (five
training embeddings); generation reads no holdout pose or image. May 2 stays
out of training, pseudo views, priors and setting choices. No lidar tuning.

| | 5 views | 5 views + recipe |
| --- | ---: | ---: |
| Holdout May 2 PSNR | 17.73 dB | 17.21 dB |
| Training-view PSNR | 23.98–38.08 dB | 21.48–31.02 dB |
| Registered MAE, all labelled | 2.82 m | 2.78 m |
| ground / buildings / vegetation | 2.77 / 2.87 / 2.78 m | 2.30 / 3.36 / 2.19 m |
| Registered RMSE / NMAD | 4.17 / 2.82 m | 5.09 / 1.47 m |
| Registered completeness | 99.6% | 99.8% |
| Horizontal offset from lidar | 1.6 m (1.5 W, 0.5 N) | 1.8 m (1.5 W, 1.0 N) |
| Unregistered median height error | −0.92 m | −1.92 m |
| Splats | 107,464 | 134,252 |
| Train time, 30k steps | 10.7 min | 42.5 min |
| Peak allocated tensor memory | 214.3 MiB | 2,708.7 MiB |

**Retain the five-view baseline in Cesium.** The rule fixed before the run
requires at least 0.2 m improvement in all-labelled registered MAE and no class
regression above 0.3 m. The measured improvement is only 0.035 m; buildings
worsen by 0.487 m. Both gates fail, so no recipe tiles or viewer switch were
made. Degree-1 SH is supported by the tiler's existing linear SH bake.

Both DSM and evaluation comparisons were reviewed against the baseline. Ground
and roof surfaces look smoother, with less streaking, but false high structure
persists around the large cross-shaped building and its courtyard. RGB renders
are softer, with visible colour changes, including a greener holdout. Lower
NMAD alongside higher building MAE and RMSE means the central error spread
shrinks while large errors remain costly. The reference-data gain does not
transfer to this five-view pilot as a useful overall height improvement.

The known anchor makes the registration shift a placement measurement here:
1.5 m west, 1.0 m north and 0.68 m low. The unregistered median is 1.92 m low;
median and fitted vertical offset differ because the errors are asymmetric.
Appearance-aware export passes, with 1.44e-9 m transform round trip and
≥113.17 dB fused/raw render equivalence using the fixed appearance embedding.
No holdout appearance fitting. One block, lidar date unknown; no broader
accuracy or production-readiness claim. Next: Web app v1 (viewer + API).

Checks: 69 SfM/Python tests pass (two pyproj-dependent tests skip there and
pass in the imagery environment); 34 imagery tests, four reference-adapter
tests, Ruff E/F and four npm tests pass. Viewer files and browser thresholds
are unchanged; conditional recipe tiling/browser checks were not needed.

## Nineteen training views

### 2026-10-30: export unblocked, recipe passes

The removal policy was fixed before inspecting removal statistics: remove ENU
centres beyond the existing 350 m horizontal envelope before fusion, with a
hard failure above **1% of total activated raw opacity mass**. This limits the
removal to a small opacity share; every existing geometry/transform gate stays
unchanged. Basic removes 38 splats (mass fraction 9.9382471e-8, radius
350.283–396.348 m); recipe removes 16 (2.5247163e-8, 351.818–375.878 m).
Fused-versus-raw-subset render equivalence uses the same geometry, 3D filter and
learned per-splat appearance embeddings: minimum 97.18/120 dB. The five-view
re-export removes zero; PLY bytes, render-equivalence record and round trip
match the original exactly. Existing run/export files were preserved.

| Run | MAE all | Ground | Building | Vegetation | Holdout PSNR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Five-view baseline | 2.817 m | 2.768 | 2.872 | 2.777 | 17.73 dB |
| 19-view basic | 3.027 m | 3.179 | 2.911 | 2.465 | 16.64 dB |
| 19-view recipe | 1.359 m | 1.396 | 1.304 | 1.526 | 15.49 dB |

The recipe has the lower registered all-labelled MAE and passes the unchanged
rule: 1.458 m improvement (required >=0.2 m), with ground/building/vegetation
changes of -1.372/-1.568/-1.251 m (no regression >0.3 m). Basic fails.
Holdout PSNR is the existing trained-model evaluation; no holdout-based choices
or training runs were made. Lidar flight date remains unknown.

New exports: basic `export-r3`, recipe `export-r2`; depth/lidar evidence is in
`artifacts/validation/pilot-2015-19v-01-depth-r3` and
`artifacts/validation/pilot-2015-19v-recipe-01-depth-r2`. The first sandbox GPU
attempt left a partial basic `export-r2`, preserved without modification.
The winning recipe was tiled with the unchanged five-view
`configs/tiles-2015-pilot.json` profile: 66,606 owned splats, 21 tiles,
1,586,904 GLB bytes. Published `jax068-2015-19v-r1` with `make_active=False`;
all active pointers remain unchanged, including `jax068-2015-5v-r2`.

Validation: export/evaluation units 14 passed, 1 skipped (torch absent in SfM);
SfM/Python 100 passed, 5 skipped; imagery 34 passed. The tensor-subset invariants
also pass a separate CPU check using stage1 torch. Both GPU exports and depth
renders passed sequentially. Commands, hashes and full registered metrics:
[result-2015-19v.json](result-2015-19v.json). Tile evidence:
[tiles-2015-19v.json](tiles-2015-19v.json).

### 2026-10-29: original training and blocked-export record

The following records the original attempt before the follow-up above.

The approved [selection](../configs/jax068-pilot-19v.json) uses all 19 usable
training acquisitions from October 5, 2014 to December 21, 2015 (442.007 days),
with May 2 held out. All 20 RGB crops were inspected before SfM; none was
rejected. No clouds, haze or obvious RGB misregistration were visible. Winter
building shadows are pronounced around the central courtyard but localized;
roofs and street features remain readable. Seasonal tint, vegetation, vehicles
and illumination vary. Per-view gates remain unchanged.

All 40 PAN/MSI sources were verified (28.83 GiB total, 19.65 GiB newly fetched).
The initial parallel feature extraction failed the image/camera ID check;
single-thread extraction preserves the ordering required by the pinned
initializer. The fresh reconstruction registered all 19 training views with
23,931 points and 85,988 observations; median/p95/max residuals are
0.408/1.481/2.615 px, with tracks of 2–19 views. The holdout is absent from the
COLMAP database. Loader verification and GPU preflight pass.

Both profiles were committed before evaluation (`6f7dbaa`): basic Stage 1 and
the unchanged authors' recipe, each at resolution 2, 200k splat doubling cap
and 4.5 GiB allocation budget. Both time limits are two hours; the previous
30-minute basic limit was close to the 26-minute 17-view reference run.
No holdout or lidar tuning. Compact evidence:
[result-2015-19v.json](result-2015-19v.json).

| Metric | 3v basic | 5v basic | 5v recipe | 19v basic | 19v recipe |
| --- | ---: | ---: | ---: | ---: | ---: |
| Holdout PSNR (dB) | 17.57 | 17.73 | 17.21 | 16.64 | 15.49 |
| Registered MAE all (m) | 3.32 | 2.82 | 2.78 | — | — |
| Ground (m) | 2.93 | 2.77 | 2.30 | — | — |
| Buildings (m) | 3.70 | 2.87 | 3.36 | — | — |
| Vegetation (m) | 3.75 | 2.78 | 2.19 | — | — |
| Registered RMSE (m) | 5.04 | 4.17 | 5.09 | — | — |
| Registered NMAD (m) | 2.99 | 2.82 | 1.47 | — | — |
| Completeness | 99.0% | 99.6% | 99.8% | — | — |
| Horizontal offset (m) | 1.8 | 1.6 | 1.8 | — | — |
| Splats | 102,543 | 107,464 | 134,252 | 114,500 | 119,248 |
| Training time (min) | 9.0 | 10.7 | 42.5 | 11.7 | 42.1 |
| Peak allocated memory (MiB) | 214.9 | 214.3 | 2,708.7 | 239.3 | 2,636.1 |

Basic training completed 30k steps and checkpoint/resume validation, but export
failed the existing ENU extent gate: 38 splat centres exceed the 350 m
horizontal radius, reaching 396.35 m. The full centre bbox spans
616 × 660 × 144 m. Attributes are finite and transform round trip passes
(1.44e-9 m), but these do not supersede the extent gate. Export failure stopped depth rendering and lidar
validation; dashes denote unmeasured values. No extent gate was relaxed.
The RGB comparison was reviewed: recognizable building/road structure,
cross-date colour mismatch and border smearing.

The recipe also completed 30k steps and checkpoint/resume validation, but its
export fails the same radial gate: 16 centres exceed 350 m, reaching 375.88 m;
the bbox spans 565 × 581 × 124 m. The violating splats have low opacity
(basic 0.000096–0.000576; recipe about 0.000097), but the existing gate applies
to every centre. No pruning or gate change was introduced after evaluation.
The recipe's RGB comparison was reviewed: smoother, softer surfaces and warmer
winter colours, with border smears remaining. Its holdout PSNR is 15.49 dB.
Neither run produced a depth DSM or `dsm-comparison.png`; render equivalence
was not reached because extent validation precedes it.

The fixed decision remains: lower registered all-labelled MAE among the two
runs, at least 0.2 m better than five-view basic, with no class more than
0.3 m worse. Neither run establishes the lidar rule, so no 19-view tiling,
release or activation was performed. The five-view model stays active
(`jax068-2015-5v-r2`, published by concurrent app work using the same five-view
run and quality evidence). More views alone did not improve
held-out appearance here; the wider seasonal span and basic model's lack of
appearance correction are confounding factors. Whether pipeline cameras and
this recipe reach about 1.4 m MAE remains unanswered because export blocked
height measurement. One block, lidar date unknown; no production-readiness claim.

Checks: 89 SfM/Python tests pass (two pyproj-dependent skips), 34 imagery,
12 API, four root JavaScript and eight web unit tests pass. App and old pilot
browser tests pass; this experiment changed no expectations or thresholds. Screenshots
were reviewed. Ruff E/F passes on `pipeline/reconstruct.py`.
Next: Phase 3: two neighbouring blocks (shared ownership grid). The export
extent blocker remains recorded; the more-views experiment is complete with
height comparison blocked.

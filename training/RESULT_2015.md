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
- The tileset root transform is `model_to_ecef` in column-major order, stored
  in the manifest. Splats stay in Skyfall's Z-up model frame; the tiler owns
  any glTF Y-up conversion and must apply it exactly once.
- Tiler rules: `REPLACE` refinement only. Child geometric error is non-zero
  and strictly below the parent's. Every tile and all its ancestors must
  contain each splat's centre plus about 3σ of its extent. Placement uses
  `root.transform`, not `CESIUM_RTC`.

Next milestone: crop to the block, tile into `KHR_gaussian_splatting` + SPZ
3D Tiles and check placement in CesiumJS.

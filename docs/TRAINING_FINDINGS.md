# Stage 1 training findings

Updated 2026-10-29. This page condenses the height experiments on the Jacksonville
JAX_068 block into guidance for later product training runs. The measured
details, hashes and commands are in [the Stage 1 result](../training/RESULT_2015.md).

All heights are compared with the DFC2019 JAX_068 airborne lidar DSM using a
rendered-depth DSM. MAE is measured after `dsmr` registration, so it scores
surface shape. The lidar flight date is unknown, and everything here comes from
one 250 m block. Treat these as directions, not guarantees.

## Results

| Run | Training views | Recipe | Holdout PSNR | MAE all | Ground | Buildings | Vegetation | RMSE |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pipeline `pilot-2015-04` | 3 | Basic Stage 1 | 17.57 dB | 3.32 m | 2.93 | 3.70 | 3.75 | 5.04 |
| Pipeline `pilot-2015-8w-01` | 5 | Basic Stage 1 | 17.73 dB | 2.82 m | 2.77 | 2.87 | 2.78 | 4.17 |
| Pipeline `pilot-2015-8w-depth-01` | 5 | Static MoGe priors on satellite images | 15.18 dB | 7.83 m | 7.48 | 8.51 | 3.94 | 10.66 |
| Pipeline `pilot-2015-8w-recipe-01` | 5 | Authors' Stage 1 recipe | 17.21 dB | 2.78 m | 2.30 | 3.36 | 2.19 | 5.09 |
| Pipeline `pilot-2015-19v-01` | 19 | Basic Stage 1 | 16.64 dB | — | — | — | — | — |
| Pipeline `pilot-2015-19v-recipe-01` | 19 | Authors' Stage 1 recipe | 15.49 dB | — | — | — | — | — |
| Authors' data `ref-jax068-01` | 17 | Basic Stage 1 | 21.35 dB | 2.36 m | 2.67 | 2.06 | 1.93 | 4.03 |
| Authors' data `ref-jax068-recipe-01` | 17 | Authors' Stage 1 recipe | 18.90 dB | **1.38 m** | 1.24 | 1.52 | 1.62 | 2.43 |

The 19-view runs trained successfully but failed the existing 350 m radial
ENU export envelope (maximum centre radius 396.35 m basic, 375.88 m recipe).
Depth/lidar validation stopped; dashes mean unmeasured, not a regression score.
The five-view release stays active because neither run establishes the fixed
lidar decision rule. [Details](../training/RESULT_2015.md#nineteen-training-views).

- **Basic Stage 1** uses the pilot settings: resolution 2, SH 2, no appearance
  model, no depth losses.
- **Authors' recipe** is the Skyfall README Stage 1 command: live MoGe
  pseudo-view depth (`--lambda_pseudo_depth 0.5`), appearance model, opacity
  weight 10, SH 1. Resolution 2 and a 200k splat cap fit it to a 6 GB GPU.
- **Holdout PSNR** compares different frames: May 2 for the pipeline runs and
  the authors' two test frames for the 17-view data. Compare it only within
  each dataset.
- **Authors' data** uses the authors' cameras and an assumed ENU anchor. Its
  shape metrics are valid; its placement offsets are not.

## Findings

1. **View count is the main lever.** MAE fell steadily from 3 to 5 to 17
   views (3.32 → 2.82 → 2.36 m), and buildings gained the most. The 17-view
   run also used the authors' cameras, so not all of that gain comes from
   view count alone.
2. **The authors' recipe helped with the authors' 17-view data, but not the five-view pilot.**
   - With 17 views it cut MAE by 41% (2.36 → 1.38 m), improving every class.
   - With 5 views it made almost no difference overall (2.82 → 2.78 m) and
     made buildings worse (2.87 → 3.36 m), while ground and vegetation
     improved.
   - The likely reason: the recipe renders the current model from synthetic
     viewpoints and asks MoGe for depth. With few views, the buildings are
     still smeared needle-like splats, so those renders and MoGe's depth are
     poor.
   - The pipeline's own 19-view height comparison remains blocked at export;
     it does not yet verify that recipe gain. Holdout PSNR fell to 16.64 dB
     basic and 15.49 dB recipe across the 442-day acquisition span.
3. **Do not use static MoGe priors on the satellite images**
   (`--lambda_depth`). The authors leave this off.
   - MoGe values were about 1,900–7,800 units with only about 0.2 units of
     variation inside each image.
   - The Pearson loss sets padded pixels to 0, so it mostly matched "inside
     the image" against "padding". That flattened the surface and degraded
     colour.
   - On valid pixels, MoGe agreed with the baseline depth only on some views
     (correlation −0.13 to 0.97).
4. **The recipe trades appearance for shape.** On the 17-view data, holdout
   PSNR fell 21.35 → 18.90 dB while MAE improved. Renders are smoother and
   shift colour slightly. Choose by product goal, and record both.
5. **Placement is independent of the recipe.** On the pipeline data, the
   horizontal offset from the lidar stayed 1.6–1.8 m (no ground control) and
   vertical within about 1 m. More views and better recipes improve shape,
   not absolute placement.
6. **The laptop GPU is enough for a block.**

   | Run | Peak allocated | Time for 30k iterations |
   | --- | ---: | ---: |
   | Basic, 5 views | 0.2 GiB | 11 min |
   | Basic, 17 views | 0.5 GiB | 26 min |
   | Recipe, 5 views | 2.65 GiB | 42.5 min |
   | Recipe, 17 views | 2.9 GiB | 53.4 min |
   | Basic, pipeline 19 views | 0.23 GiB | 11.7 min |
   | Recipe, pipeline 19 views | 2.57 GiB | 42.1 min |

   All runs used a 6 GB RTX 3060 Laptop GPU at resolution 2.

## Recommended setup for product training

1. **Collect about 15–20 training views per block**, plus at least one whole
   acquisition held out.
   - Prefer wide azimuth and elevation spread.
   - A wider date window is acceptable for more views, but screen for
     construction, vegetation and shadow changes.
   - Keep the per-view gates: cloud ≤10% on the block, coverage, off-nadir
     ≤40°, sun elevation ≥15°.
2. **Use the authors' Stage 1 recipe** (`stage1-pseudo-depth` profile,
   appearance model on) once there are about 15 or more views. Below that,
   use basic Stage 1 and expect weak building heights.
3. **Keep `--lambda_depth 0`.** Use MoGe only through pseudo views.
4. **Fix settings before evaluation**, and gate the result against the
   current baseline with predeclared thresholds, as the pilot did: at least
   0.2 m MAE gain and no class more than 0.3 m worse.
   - Never tune on the holdout or the lidar.
5. **Validate every block** with the rendered-depth lidar check where lidar
   exists. Report shape (registered MAE per class) and placement (registration
   shift) separately.
6. **Expect placement errors of 1–2 m without ground control.** Fix them, if
   needed, as a separate, versioned georeference adjustment, not through
   training.

## Open questions

- Whether pipeline data reaches about 1.4 m MAE with its own cameras and a
  known anchor remains unanswered: both 19-view exports fail the existing
  extent gate before rendered-depth lidar validation. Any future treatment of
  low-opacity outliers needs a declared policy before another evaluation.
- Where the recipe starts to help: somewhere between 5 and 17 views.
- How much seasonal and building change a wider date window adds.
- Whether full resolution (resolution 1) or Stage 2 refinement on a larger
  GPU improves buildings further.

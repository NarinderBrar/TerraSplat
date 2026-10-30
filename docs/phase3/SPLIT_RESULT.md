# Phase 3 split result (2026-10-30)

West and east 19-view recipe blocks trained in 41 and about 42 minutes. Each half-covers the DFC2019 JAX_068 lidar.
Evidence: `training/phase3-split-seam-check.json`; same-area baselines:
`training/phase3-single-block-{west,east}-half-validation.json`.

## Verdict under the pre-registered rules: fail

| Check (configs/phase3-split-decision.json) | Limit | Measured | Result |
| --- | --- | --- | --- |
| Ownership: doubled centres, gap, overlap | 0 | 0 / 0 / 0 | pass |
| West block registered MAE | ≤ 1.659 m | 1.681 m | **fail** |
| East block registered MAE | ≤ 1.659 m | 1.100 m | pass |
| West seam band vs interior | ≤ interior + 0.5 m | 0.94 vs 1.74 m | pass |
| East seam band vs interior | ≤ interior + 0.5 m | 1.05 vs 1.10 m | pass |
| Raw ground height step across seam | ≤ 0.5 m | 0.74 m | **fail** |
| Raw all-pixel height step across seam | ≤ 1.0 m | 2.54 m | **fail** |

Stage B stopped before tiling and publication, as designed.

## Diagnostics (after the fact; they do not change the verdict)

Same-area comparison with the single central 19-view model, using the same grid cells and registration:

| Lidar area | Single-block model | Split block | Difference |
| --- | ---: | ---: | ---: |
| West half | 1.640 m | 1.681 m | +0.04 m |
| East half | 1.083 m | 1.100 m | +0.02 m |

The single block's 1.359 m averages a harder west half and an easier east half. The block-MAE rule compared each
half with that whole-block average.

The raw height-step rule includes real terrain and building differences between the two 10 m strips. The lidar itself
steps 0.24 m (ground) and 2.19 m (all pixels) across the same strips. The model's step beyond the lidar's is 0.33 m
(ground) and 0.14 m (all pixels).

## Lessons for the next pre-registered rules

- Compare each block with the reference model on the same area, not with a whole-block average.
- Judge seam height continuity by the step beyond the lidar's own step (residual), not the raw strip difference.
- Fix these before the next experiment's results exist; do not re-grade this run with them.

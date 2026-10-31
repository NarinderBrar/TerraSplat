# 1 km grid result (2026-10-31)

All 16 blocks of `configs/jax068-grid-1km.json` trained (21:58 on 2026-10-30 to 12:55, about 56 min each) and
published to their job folders. B3 first failed export on 30 high floaters; the floater rule was amended after that
failure and B3 reran from export with its training reused (docs/NEXT_STEPS.md).

Scored with `pipeline.grid_seams` against the pre-registered `configs/grid-1km-decision.json`; evidence in
`training/grid-1km-result/` (`seams.json`, `inputs.json`). **Result: FAILED** (2 of 4 seams fail). Thresholds are
not changed.

## Cells (registered all-labelled MAE vs the single-block model on the same lidar pixels; limit +0.3 m)

| Cell | Grid | Single block | Excess | Lidar coverage | |
| --- | --- | --- | --- | --- | --- |
| A2 | 1.412 m | 1.395 m | +0.02 m | 25.8% | pass |
| A3 | 1.202 m | 1.083 m | +0.12 m | 26.1% | pass |
| B2 | 2.042 m | 1.879 m | +0.16 m | 26.3% | pass |
| B3 | 1.097 m | 1.091 m | +0.01 m | 26.6% | pass |

## Seams (band MAE excess over interior, limit 0.5 m; residual height step, limits 0.5 m ground / 1.0 m all)

| Seam | Band excess (each side) | Ground step | All step | |
| --- | --- | --- | --- | --- |
| A2\|A3 | A2 −0.14, **A3 +0.63** | **−0.61 m** | −0.35 m | fail |
| B2\|B3 | B2 −1.34, B3 −0.42 | +0.35 m | −0.04 m | pass |
| A2\|B2 | B2 +0.35, **A2 +2.40** | **−2.19 m** | +0.55 m | fail |
| A3\|B3 | B3 +0.08, A3 +0.30 | +0.14 m | +0.16 m | pass |

No seam has doubled splat centres (ownership passes everywhere).

Notes, not changing the outcome: the A2\|B2 ground step rests on only 142 ground pixels on the B2 side; A2's 10 m band
along that seam has 3.62 m MAE against 1.22 m in its interior, so the error is concentrated at A2's southern edge.

## Consequence

Per the decision file: every cell passes and may be activated; the A2\|A3 and A2\|B2 seams fail, so the grid is
published inactive with this measurement and discussed before activation. The other 12 cells stay labelled unmeasured.

## Decision (2026-10-31)

After reviewing all 16 cells in the browser, the user decided to activate the grid with the two failing seams
(A2|A3, A2|B2) as known issues for this version; investigating and fixing A2's southern edge is deferred. All 16
releases (`jax068-2015-grid-*-r1`) were published to `artifacts/releases` and activated, and the central single
block `jax068-2015` was retired with `python -m pipeline.publish --retire-dataset jax068-2015` (its releases
`jax068-2015-19v-r1` and `jax068-2015-19v-classes-r1` moved to `artifacts/releases/.withdrawn/`).

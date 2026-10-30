# True-neighbour 3DEP qualification — 2026-10-30

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

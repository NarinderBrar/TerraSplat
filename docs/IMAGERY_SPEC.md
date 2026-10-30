# Commercial imagery specification

What to buy so the TerraSplat pipeline can reconstruct a block, and what accuracy to expect. Check any offer
before purchase with:

    .venv-sfm/bin/python -m pipeline.imagery_check offer.csv [--holdout ID] [--output report.json]

`configs/imagery-candidates.example.csv` shows the input format: one row per offered image, taken from the
vendor's catalogue metadata. No pixels are needed.

To build that CSV from delivered (or sample) metadata instead of typing it:

    python -m pipeline.vendor_metadata DELIVERY... --aoi WEST SOUTH EAST NORTH --output offer.csv

It reads Maxar `.IMD`/`.RPB` (loose or in `.tar` archives; PAN and MSI products of one catalog id are merged) and
Airbus Pléiades / Pléiades Neo DIMAP v2 (`DIM_*.XML` with `RPC_*.XML`), normalises product levels (Maxar `LV1B`/Basic
→ raw; `LV2A`/Standard and `LV3*`/Ortho → not raw) and checks each image footprint against the AOI. The Maxar reader
reproduces the recorded metadata of the 19-view selection exactly. The DIMAP reader follows the documented DIMAP v2
structure (incidence angle → elevation = 90° − incidence; viewing azimuth; centre `Located_Geometric_Values`) but has
only been tested on a synthetic file: check its output against the first real Pléiades Neo delivery.

## Required for every image

Images failing any of these are rejected by the check.

| Requirement | Value | Why |
| --- | --- | --- |
| Product level | Raw sensor geometry: L1B / Basic / Primary / Sensor. **Not** orthorectified, not a mosaic | Orthorectification removes the viewing geometry that heights are computed from |
| Camera model | RPC delivered with the image (e.g. `.RPB`, `RPC00B`, DIMAP `RPC_*.XML`) | Initial cameras come from the RPCs |
| Bands | Panchromatic **and** multispectral, as separate unsharpened products | The pipeline pan-sharpens its own RGB crops |
| Resolution | Panchromatic GSD ≤ 0.5 m | Measured results are from WorldView-3, 0.31–0.4 m |
| Coverage | Each image fully covers the AOI plus a 20 m halo | Every block uses every training view |
| Off-nadir | ≤ 40° | Same limit as `pipeline.acquisitions` |
| Sun elevation | ≥ 15° | Long shadows hide ground and facades |
| Cloud cover | ≤ 10% **over the AOI** (not whole-scene) | Ask for AOI-level cloud estimates or masks |

Also request each image's metadata file (capture time, satellite azimuth and elevation, sun azimuth and
elevation, GSD, cloud cover), full bit depth with no dynamic-range adjustment, and a licence that allows
derived 3D models to be published.

## How many images, from which directions

Height accuracy depends mainly on the number of views and how widely their directions differ. The check
compares the training views (all usable images minus one held out) with the sets we measured against lidar
(registered MAE, Jacksonville JAX_068, one 250 m block):

| Training views | Largest compass gap | One-plane gap* | Widest angle between two views | Measured MAE |
| ---: | ---: | ---: | ---: | ---: |
| ≥ 19 | ≤ 96.6° | ≤ 39.6° | ≥ 56.4° | **1.36 m** (authors' training recipe) |
| ≥ 5 | ≤ 185.2° | ≤ 76.4° | ≥ 49.6° | 2.82 m |
| ≥ 3 | ≤ 185.2° | ≤ 102.4° | ≥ 49.6° | 3.32 m |
| 2 | – | – | – | Unmeasured; expect worse than 3.3 m |
| 1 | – | – | – | Not possible: heights need at least two directions |

\* Compass directions folded onto 0–180°, so opposite views count as one direction. A single along-track pass
(forward, nadir and backward) has a one-plane gap near 180° and gets no tier above "unmeasured".

Recommendations:

- **Buy 20 images to reach the measured 1.36 m class:** 19 training views plus one held out for validation.
  Fewer views give the lower rows. With 5 training views the authors' recipe gave no gain over basic training.
- **Collect from at least two different directions**, e.g. two or more passes or satellites, not one
  along-track sequence. Ask for azimuths spread around the compass.
- **Hold one image out.** Buy at least 4 usable images so one whole acquisition can be kept out of calibration,
  training and every setting choice. The check picks the most nadir image unless `--holdout` names one.

## Capture dates

- **Same day is best:** no construction or seasonal change between views, and consistent shadows.
- **Within 7 days** is acceptable.
- **A wider span** needs screening for change before training. Our 19-view set spanned 442 days: heights still
  reached 1.36 m, but the held-out image's colours matched less well (holdout PSNR 15.5 dB vs 17.7 dB for 5 views).

## Validation data

Get an independent height reference for at least part of the AOI, ideally from the same period: airborne lidar,
or a surveyed DSM. Without it, accuracy can't be measured, only expected. An older DSM is valid only in areas
verified as unchanged.

## Limits of this specification

- The accuracy tiers come from one flat, low-rise 250 m block in one city. Dense high-rise areas are likely
  harder.
- Same-day sets of more than two images and 2-image stereo have not been measured.
- The pre-purchase check reads Maxar and Airbus DIMAP metadata (`pipeline.vendor_metadata`). Training still needs
  `pipeline.acquisitions` and `pipeline.block_inputs` to accept Pléiades Neo products (they read WorldView-3 only).

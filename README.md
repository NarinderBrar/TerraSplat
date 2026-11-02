# TerraSplat

**Satellite imagery to interactive 3D maps.**

TerraSplat turns satellite images into 3D Gaussian splats and streams them onto a web globe.
It also labels buildings, trees, roads, ground, grass, and water so you can highlight them in 3D.

[![TerraSplat showing a Jacksonville reconstruction and semantic class controls](public/photo.png)](https://www.youtube.com/watch?v=HKcTSGbMY7Q)

**[Watch the demo](https://www.youtube.com/watch?v=HKcTSGbMY7Q)** ·
**[Download the segmentation model](https://www.kaggle.com/code/npsbrar1986/terrasplat-landcover-v5)**

## What it does

- **Reconstruct:** build geographically placed 3D scenes from multiple satellite views.
- **Stream:** load the detail you need as you explore, using 3D Tiles, glTF, and SPZ compression.
- **Segment:** select scene classes and highlight them with animated holographic effects.

Built with **Python, SatelliteSfM, Skyfall-GS, SegFormer-B2, FastAPI, React, and CesiumJS**.

## Quick start

Requires **Node.js 22 LTS (22.12+) or 24+**, and a browser with WebGL2.
Try the included synthetic splat demo:

```bash
git clone https://github.com/NarinderBrar/TerraSplat.git
cd TerraSplat
npm ci
npm run dev
```

Open **http://127.0.0.1:5173**. This is a small diagnostic demo with five synthetic splats.

For the **full satellite viewer shown above**, follow the [setup guide](docs/GETTING_STARTED.md).
Scene assets are stored separately and are not included in a clone. The guide covers the Python API,
existing release catalogues, reconstruction training, and the experimental segmentation catalogue.

## Current status

Research prototype: our own **1 km² Jacksonville reconstruction** spans 16 trained blocks.
The viewer also integrates **12 externally trained Skyfall-GS reference scenes**. Experimental segmentation
layers exist for all 28 datasets in a separate review catalogue.

Current limits include historical imagery, a 6 GB training GPU, two failing grid seams, and segmentation
accuracy below some targets. See [results and limitations](docs/TECHNICAL_OVERVIEW.md#results).

## Learn more

| Guide | What you will find |
| --- | --- |
| [Getting started](docs/GETTING_STARTED.md) | Full viewer setup, pipeline commands, and tests |
| [Technical overview](docs/TECHNICAL_OVERVIEW.md) | Architecture, reconstruction stages, results, and limitations |
| [Segmentation model](docs/SEGMENTATION_MODEL.md) | Public Kaggle weights, download commands, and class prediction |
| [All documentation](docs/README.md) | Detailed engineering notes and experiments |

## Licence

TerraSplat's original code is [MIT licensed](LICENSE). Third-party code, datasets, model weights, and scene
assets have separate terms; see [data sources and terms](docs/TECHNICAL_OVERVIEW.md#data-sources-and-terms).

# Getting started

[Project home](../README.md) · [All documentation](README.md)

Choose the included synthetic demo, the full viewer with existing scene assets, or reconstruction training.
All commands run from the repository root. The full setup uses `uv` for Python environment management.

## Clone the repository

```bash
git clone https://github.com/NarinderBrar/TerraSplat.git
cd TerraSplat
```

## Try the bundled demo

```bash
npm ci
npm run dev
```

Open http://127.0.0.1:5173. This diagnostic viewer contains five synthetic splats and requires no Python API,
training checkpoint, or satellite downloads. It is different from the full web app shown in the screenshot.

## Full viewer and training setup

The full viewer needs Node.js, Python 3.12, `uv`, and an existing scene catalogue. The NVIDIA GPU, compiler,
and larger disk requirements below apply to reconstruction training.

### Requirements

| Component | Tested with |
| --- | --- |
| OS | Linux (Ubuntu) |
| GPU (training) | NVIDIA, 6 GB VRAM (RTX 3060 Laptop), CUDA 12.8 wheels; driver only, no system CUDA needed |
| Node.js | 22.12+ or 24+ |
| Python | 3.12 (pipeline, API, imagery); 3.10 (Skyfall-GS training) |
| Tools | `uv`, Git, GCC (to build the training extensions) |
| Disk | ≥ 20 GiB free per job stage; source imagery and models are several GB per block |

### Python environments

Heavy dependencies are kept apart; each environment is created by the setup steps in its document. Run Python with
`env -u PYTHONPATH` (or `export PYTHONPATH=`) if the host exports a `PYTHONPATH` (for example ROS).

| Environment | Used for | Setup |
| --- | --- | --- |
| `.venv-imagery` | Imagery, GDAL/Rasterio, lidar, semantics scoring | [imagery/README.md](../imagery/README.md) |
| `.venv-sfm` | Satellite SfM, job runner, publisher | [sfm/README.md](../sfm/README.md) |
| `.venv-stage1` | Skyfall-GS training, export, depth rendering (Python 3.10, PyTorch 2.7.1) | [training/README.md](../training/README.md) |
| `.venv-api` | FastAPI release API | [services/api/README.md](../services/api/README.md) |
| `.venv-landcover` | Land-cover model training and prediction | [docs/SEMANTICS.md](SEMANTICS.md) |

### Scene assets are required for the full viewer

The screenshot and video show the full satellite viewer. Its imagery, models, and published scene catalogues are
not bundled in Git, and a ready-to-download TerraSplat scene release is not currently provided. Cloning the code
or downloading the segmentation checkpoint alone will not reproduce those scenes.

Use an existing local release catalogue, or generate and publish one through the pipeline below. The default
catalogue must contain `artifacts/releases/catalog.json` and its referenced release and tile files. A copy must
include the complete published catalogue, not just the PLY model. If `artifacts` is a broken local symlink, restore
its target or point `TERRASPLAT_RELEASES_DIR` to a complete catalogue elsewhere.

### Run the web app on a published catalogue

```bash
npm ci
npm --prefix apps/web ci
uv venv --python 3.12 .venv-api
uv pip install --python .venv-api/bin/python -r services/api/requirements.txt
npm run dev:app            # API on :8000, web app on http://127.0.0.1:5173
```

To view the experimental segment layers, serve the review catalogue:

```bash
TERRASPLAT_RELEASES_DIR=artifacts/releases-grid-review npm run dev:app
```

The default command serves `artifacts/releases`; review releases have not been promoted there.
`TERRASPLAT_RELEASES_DIR=<dir> npm run dev:app` serves another catalogue. On hybrid Intel/NVIDIA laptops, start
Chrome on the NVIDIA GPU for full quality ([apps/web/README.md](../apps/web/README.md#running-chrome-on-the-nvidia-gpu-linux-hybrid-graphics-laptops)):

```bash
__NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only google-chrome \
  --user-data-dir=/tmp/chrome-nvidia --use-angle=vulkan \
  --enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan http://127.0.0.1:5173/
```

## Running the pipeline

A job file describes one block: catalogue, approved acquisitions, training and tiling profiles, optional grid cell,
optional lidar truth, and release identity. Examples are in [configs/jobs](../configs/jobs).

```bash
export PYTHONPATH=
.venv-sfm/bin/python -m pipeline.job configs/jobs/grid-1km-a1.json --dry-run    # print commands and reuse decisions
.venv-sfm/bin/python -m pipeline.job configs/jobs/grid-1km-a1.json --no-gpu     # CPU stages only
.venv-sfm/bin/python -m pipeline.job configs/jobs/grid-1km-a1.json              # full run (about 1 h of GPU)
.venv-sfm/bin/python -m pipeline.job configs/jobs/grid-1km-a1.json \
  --from publish --publish-to artifacts/releases --activate                     # publish to the live catalogue
```

- **Reuse:** rerunning the same command reuses every stage whose inputs, code and outputs still hash the same.
- **`--from <stage>`** forces that stage and everything after it; **`--until <stage>`** stops after it.
- **State:** `job-state.json` records each stage's command, environment, timing, log and hashes, written atomically
  under a job lock. Old attempts are kept.

## Releases and API

```text
artifacts/releases/
  catalog.json                       authoritative list of datasets, releases and active pointers
  index/<region>/tileset.json        3D Tiles index of active releases (external tilesets)
  <release_id>/release.json          manifest: provenance, footprint, quality, evidence, hashes
  <release_id>/tileset.json, tiles/  SPZ glTF 3D Tiles
```

```bash
.venv-sfm/bin/python -m pipeline.publish --config configs/releases/<id>.json --releases artifacts/releases [--activate]
.venv-sfm/bin/python -m pipeline.publish --releases artifacts/releases --activate-release <release_id>   # switch / roll back
.venv-sfm/bin/python -m pipeline.publish --releases artifacts/releases --retire-dataset <dataset_id>    # files kept in .withdrawn/
```

Release IDs are immutable; a rebuild needs a new ID. Manifests follow the JSON Schemas in [schemas/](../schemas). The
read-only [API](../services/api/README.md) serves datasets, releases, quality, footprints (GeoJSON) and the regional
index; OpenAPI docs are at `/docs`. Details: [docs/RELEASES.md](RELEASES.md).

## Web app

[apps/web](../apps/web/README.md) (React, Vite, TypeScript, CesiumJS 1.146):

- regional index with dataset and release selection, footprints and quality panels;
- level-of-detail control: full detail for the nearest blocks within 2 km, with hysteresis to stop LOD flipping;
- reveal effects: coarse tiles as points, leaves revealed by scan lines, parents held until the reveal ends
  ([docs/REVEAL_EFFECTS.md](REVEAL_EFFECTS.md));
- GPU-aware quality profiles (integrated GPUs get a reduced profile; override with `?quality=full|reduced`);
- class selection: highlight splats from a per-release sidecar layer; buildings use TerraScope's blue holographic
  grid, scans and trails, blended at 20%–40% with the base colour (steady 30% with reduced motion or reduced quality)
  ([docs/SEGMENTS.md](SEGMENTS.md));
- base imagery: bundled Natural Earth II plus Esri World Imagery (disable with `VITE_REMOTE_IMAGERY=false`); no
  Cesium ion services.

## Testing

```bash
npm test && npm run test:browser                     # pilot viewer (Node test runner, Playwright)
npm run test:app                                     # web app unit + browser tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-sfm/bin/python -m pytest tests/sfm tests/python -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-imagery/bin/python -m pytest tests/imagery -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-api/bin/python -m pytest tests/api -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-landcover/bin/python -m pytest tests/landcover -q
```

All Python commands assume an empty `PYTHONPATH`.

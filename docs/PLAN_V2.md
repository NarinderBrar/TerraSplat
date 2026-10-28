# TerraSplat plan v2

Date: 2026-10-28. Status: Phase 0 implementation started; see [PHASE0.md](PHASE0.md) for executed checks and remaining gates.

Current implementation scope: pipeline tooling and a minimal validation viewer. Product web app features are deferred.

This revision builds on [PLAN.md](PLAN.md) and [RESEARCH.md](RESEARCH.md). Those documents remain the reference for coordinate math, tile/LOD design rules, semantics policy and acceptance gates. This file records what changed: existing local work to reuse, the chosen imagery, hardware limits and the renderer decision.

## 1. Prototype goal

**1 km² of Jacksonville, from raw WorldView-3 images to georeferenced Gaussian splat 3D Tiles that load on the CesiumJS globe automatically, checked against DFC2019 lidar heights.**

## 2. Decisions

| Topic | Decision | Reason |
| --- | --- | --- |
| Imagery | CORE3D public WorldView-3, Jacksonville | Free, no login, ~0.31 m panchromatic, 26 multi-date views with RPCs |
| Validation data | DFC2019 / US3D lidar DSMs for Jacksonville | Same city; independent height and placement check (IEEE DataPort login, free) |
| Delivery format | 3D Tiles + glTF `KHR_gaussian_splatting` + SPZ compression | Open standard; renderer-independent; "load directly in a map" |
| First viewer | CesiumJS (pinned version) | Globe, terrain, imagery and 3D Tiles streaming built in |
| Later viewer (optional) | WebGPU viewer reading the same tiles | Only if Cesium frame rate fails the target |
| Stage 2 generative refinement | Off | Invents detail; 6 GB VRAM cannot run it |

The data format is the contract, not the viewer. Any renderer that reads the tileset must show the same placement.

## 3. Existing local work to reuse

`TerraScope/` already solved parts of this pipeline in a non-standard format (PlayCanvas SOG + `lod-meta.json`, flat Mercator placement):

| TerraScope asset | Use here |
| --- | --- |
| `docs/splat-registration/*.json`: visual registration of all 12 Skyfall scenes (held-out max residual about 1.3–3.8 m) | Phase 1 placement for existing scenes; comparison baseline for automatic placement |
| `docs/SPLAT_STREAMING.md`: 3-level LOD (6.25% / 25% / 100%), merge via `@playcanvas/splat-transform`, continuity check < 0.15 units | LOD merge approach and validation pattern |
| `scripts/verify-splat-lod-alignment.py` | Adapt for tile-to-source continuity checks |
| LangSplat semantic fields (512×512×64 per scene) and `prepare_segmented_view.py` | Starting point for semantic voxel tiles |

TerraScope places scenes with a 2×2 EPSG:3857 transform plus a ground plane. Convert that to an ENU anchor (remove Mercator scale, handle Z-up/Y-up) before using it in a tileset.

## 4. Automatic geo-referencing

SatelliteSfM writes `enu_observer_latlonalt.json`; Skyfall's `evaluate_gs_geometry.py` expects it at `outputs_srtm/`. The published Skyfall dataset omits it, which is why TerraScope had to register visually. Running SatelliteSfM ourselves and retaining that file, plus Skyfall's normalization (`R`, `T`, `scale = 256 / radius` in `scene/dataset_readers.py`), makes placement exact:

```
tileset.root.transform = eastNorthUpToFixedFrame(lon, lat, alt) · inverse(Skyfall normalization)
```

Record this chain in `georeference.json` (see PLAN.md §5). Validate with DFC2019 lidar DSMs using Skyfall's `dsmr.py`, and compare against TerraScope's registration where the AOI overlaps an existing JAX scene.

## 5. Imagery details

Public bucket (no credentials, `--no-sign-request`):

```
s3://spacenet-dataset/Hosted-Datasets/CORE3D-Public-Data/Satellite-Images/
  Jacksonville/WV3   289 files, ~133 GB (26 PAN P1BS NITF + MSI)
  Omaha/WV3          473 files
  UCSD/WV2, Richmond/WV2, Tampa/WV2
```

Do not download everything. Choose an AOI of about 1 km², then read only that window from each PAN image. GDAL `/vsicurl/` windowed reads of these NITF files are untested; if they are impractical, download PAN files only. Crops must update RPC offsets. Pan-sharpen MSI (~1.24 m) to ~0.3 m for colour. Choose the AOI to overlap an existing Skyfall JAX scene.

## 6. Hardware budget

Local GPU: NVIDIA RTX 3060 Laptop, 6 GB VRAM.

| Stage | Fits? | Plan |
| --- | --- | --- |
| SatelliteSfM | Yes, mostly CPU | Local |
| Skyfall Stage 1 | Tight | ~250×250 m training blocks with halos, downsampled images, ~1–1.5M splat cap per block |
| Skyfall Stage 2 | No | Skip |
| LangSplat SAM + CLIP | Tight | CPU offload (already used in TerraScope runs) |
| Tiling, SPZ, validation | Yes, CPU | Local |

1 km² ≈ 16 blocks run sequentially. City scale needs a rented 24 GB+ GPU; measure one block first.

Local system RAM is 15 GB. Skyfall submodules (rasterizer, `simple-knn`, `FlowEdit`, `MoGe`, `fused-ssim`) are not checked out in `Ref/Skyfall-GS`; run `git submodule update --init` and build the CUDA extensions before any training.

Stage 2 cannot run locally at all: FlowEdit is missing, and FLUX (~12B parameters, ~24 GB in bf16, plus a ~9 GB text encoder) exceeds both VRAM and system RAM. Even quantized with offloading it would take days per scene. If ever wanted, run it on a 48 GB+ GPU and publish the result as a separate "synthesized" layer.

### Compute options

Checked 2026-10-28. Offers and prices change; recheck before use.

| Option | GPU | Cost | Use for |
| --- | --- | --- | --- |
| Local laptop | RTX 3060 Laptop, 6 GB | Free | Setup, small tests, tiling, validation |
| Kaggle Notebooks | T4 16 GB (often 2×) or P100 16 GB | Free, ~30 GPU h/week, sessions ~9–12 h | Stage 1 at higher resolution, LangSplat |
| Google Colab (free) | Usually T4 16 GB | Free, not guaranteed | Backup for Kaggle |
| [OVHcloud](https://www.ovhcloud.com/en-in/public-cloud/gpu/) | L40S 48 GB ₹128.5/h; A10 24 GB ₹70/h; L4 24 GB ₹69/h; V100S 32 GB ₹72/h; RTX 5000 16 GB ₹33/h; H100 80 GB ₹248/h | ₹18,000 new-project credit, valid 1 month from activation | Heavy runs: full AOI Stage 1, optional Stage 2 (~140 L40S hours) |
| [Alibaba Cloud PAI-DSW](https://help.aliyun.com/en/pai/create-and-manage-dsw-instances) | A10 24 GB (`ecs.gn7i-c8g1.2xlarge`) or V100 16 GB (`ecs.gn6v-c8g1.2xlarge`) | Free trial; quota unconfirmed (a blog reports 5000 CU for 3 months, unverified) | Extra Stage 1 / LangSplat runs |

Caveats:

- OVH: needs a saved payment method; new accounts often need a GPU quota increase (allow 1–2 days). Usage beyond the credit is charged. The page does not say whether prices are per GPU or per instance.
- Alibaba: running instances switch to pay-as-you-go automatically when the free quota ends. Stop them after each run.
- Free notebook sessions end without warning; checkpoint to persistent storage often. The pipeline's resumable stages are designed for this.

Sequence: laptop + Kaggle until one block runs end to end. Then activate OVH (the 1-month clock starts at activation) and run the full AOI on the L40S. Use the Alibaba trial as extra capacity.

## 7. Renderer notes

- CesiumJS is WebGL2-only; no WebGPU renderer as of October 2026.
- Cesium sorts splats with a WebAssembly radix sort on CPU web workers (`GaussianSplatSorter.js`), not a GPU compute sort. LOD bounds splat counts, which is acceptable for the prototype.
- GPU sort alternatives if needed later: [gaussian-splatting-webgpu](https://github.com/MarcusAndreasSvensson/gaussian-splatting-webgpu), [wgpu_sort](https://docs.rs/wgpu_sort/), [WebSplatter](https://arxiv.org/html/2602.03207v1).
- three.js fallback: [3D-Tiles-RendererJS-3DGS-Plugin](https://github.com/WilliamLiu-1997/3D-Tiles-RendererJS-3DGS-Plugin) for `Ref/3DTilesRendererJS`.

## 8. Pipeline

```
CORE3D WV3 PAN/MSI + RPCs (STAC record)
 → Preflight: overlap, angles, clouds, dates          ── fail → imagery/terrain only
 → AOI crop (RPC-aware) + pan-sharpen
 → SatelliteSfM → cameras, sparse points, enu_observer_latlonalt.json
 → Skyfall Stage 1 per block (halo) → fused PLY + normalization record
 → georeference.json + DSM report (DFC2019 lidar)
 → LangSplat features → class voxels / per-splat class IDs
 → Tiler: quadtree → LOD merge → SPZ → glb → tileset.json
 → Validate (3d-tiles-validator, LOD continuity, DSM error) → immutable release
 → CesiumJS app: splat tiles + semantic layer + imagery/terrain + query API
```

Tiler: write a small Python tool (adaptive quadtree, splat-transform-style merging for parents, nianticlabs/spz encoding, glb wrapping). Third-party tilers stay evaluation-only (RESEARCH.md).

Semantics as tiles: start with a 3D Tiles voxel tileset rendered by Cesium `VoxelPrimitive` (closest to the existing voxel fields). Per-tile class-ID sidecars aligned with splat order are the alternative, since SPZ cannot carry custom attributes. Keep 512-D LangSplat features server-side for text queries.

## 9. Phases

| Phase | Deliverable | Effort |
| --- | --- | --- |
| 0 | Pin CesiumJS, SPZ and extension versions; tiny test glb renders with correct placement, scale and orientation | days |
| 1 | Convert TerraScope's 12 registered scenes to 3D Tiles; view on the Cesium globe | 1–2 weeks |
| 2 | Full pipeline on one block of the Jacksonville AOI with the ENU anchor kept; compare with TerraScope registration and lidar | 2–4 weeks |
| 3 | All blocks of the 1 km² AOI; no seams or doubled opacity between tiles or LOD levels | 2–3 weeks |
| 4 | Semantic voxel tiles + query API | 2–3 weeks |
| 5 | Job runner (resume/retry), versioned releases, catalogue UI | 3+ weeks |

Estimates are planning guesses, not measurements. Stack and layout follow PLAN.md §8.

## 10. First tasks

1. Phase 0 test glb in a pinned CesiumJS build. Initialize Skyfall submodules and build its CUDA extensions.
2. Test a windowed `/vsicurl/` read from one Jacksonville PAN NITF.
3. Pick the 1 km² AOI overlapping an existing JAX scene; request DFC2019 access.

## Sources

- [CORE3D open repository](https://github.com/pubgeo/core3d-open)
- [US3D / DFC2019 paper](https://arxiv.org/pdf/1811.08739)
- [Urban Semantic 3D dataset](https://ieee-dataport.org/open-access/urban-semantic-3d-dataset)
- [CesiumJS GaussianSplatSorter.js](https://app.unpkg.com/@cesium/engine@26.1.0/files/Source/Scene/GaussianSplatSorter.js)
- [Cesium releases, April 2026](https://cesium.com/blog/2026/04/01/cesium-releases-in-april-2026)

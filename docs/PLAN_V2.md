# TerraSplat plan v2

Updated: 2026-10-29. Status: Phase 0 compatibility fixture passes; Phase 2 proceeds with the approved 2015 pilot. See [PHASE0.md](PHASE0.md) and [PHASE2.md](PHASE2.md). Phase 1 remains incomplete.

Current implementation scope: pipeline tooling and a minimal validation viewer. Product web app features are deferred.

This revision builds on [PLAN.md](PLAN.md) and [RESEARCH.md](RESEARCH.md). Those documents remain the reference for coordinate math, tile/LOD design rules, semantics policy and acceptance gates. This file records what changed: existing local work to reuse, the chosen imagery, hardware limits and the renderer decision.

## 1. Prototype goal

**Reconstruct the approved 250×250 m Jacksonville block from April–May 2015 CORE3D imagery, validate the pipeline, and export georeferenced splat tiles for CesiumJS.** Scale toward 1 km² only after the one-block checks pass.

**Latest user decision (2026-10-29): use the 2015 data for the end-to-end historical demonstration.** Resume from the prepared crops and initial cameras, keeping the three training acquisitions and May 2 holdout. The approved 13-day acquisition-span exception applies to this pilot. Label outputs with their 2015 capture dates.

Recent imagery qualification is deferred and does not block this demonstration. For a future current-city release, separately qualify recent inputs and date-compatible validation. The preceding-12-month freshness target remains provisional for that later work.

## 2. Decisions

| Topic | Decision | Reason |
| --- | --- | --- |
| Active pilot imagery | Approved April 19–May 2, 2015 CORE3D WorldView-3 group, Jacksonville | Complete the historical end-to-end pipeline using already prepared inputs |
| Current-city imagery | Deferred; no recent dataset selected | A future current-city release requires separate input qualification |
| Validation data | Independent DSM/lidar/control appropriate to the 2015 pilot; exact epoch, coverage and datum still require verification | Avoid treating scene changes or datum differences as reconstruction error; DFC2019/US3D references are candidates, not accepted ground truth |
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

Record this chain in `georeference.json` (see PLAN.md §5). For the historical benchmark, evaluate suitable DFC2019 lidar DSMs using Skyfall's `dsmr.py` and compare against TerraScope's registration where the AOIs overlap. For recent imagery, verify reference acquisition dates and use date-compatible evidence or documented unchanged checkpoints.

## 5. Imagery details

### Active historical pilot

Use the existing 250 m block with a 20 m halo, three training acquisitions (April 19, April 26 and May 1, 2015), and the May 2 holdout. Crops, RGB preparation and RPC-derived initial cameras already exist. Next resolve the SfM runtime, complete matching/bundle adjustment and sparse reconstruction, and preserve the ENU origin and pixel/normalization transforms through training. Keep the holdout out of calibration and fitting.

### Recent imagery qualification — deferred

Search constraints confirmed on 2026-10-29: free sources only; a future recent pilot may change location; no research affiliation or existing programme access. The [source qualification report](../imagery/qualification/2026-10-29/README.md) retains the initial findings. No recent dataset qualified. The subsequent decision to use 2015 imagery supersedes that search as the immediate priority; the following gate applies only when current-city work resumes.

1. Confirm the AOI, desired capture window and maximum acceptable age. Use the preceding 12 months as the proposed freshness target, subject to explicit dataset selection.
2. Check source availability, native GSD, overlapping viewing geometry, RPC delivery, cloud coverage, matched color inputs, usage/derivative rights and acquisition/processing cost. Do not assume a recent map baselayer supplies reconstructable sensor imagery.
3. Prefer a group within seven days. A wider span needs documented change screening and dataset-specific approval; the 13-day exception was for the 2015 pilot only. Freshness and within-group time span are separate checks.
4. Reserve whole acquisitions as holdouts before calibration/appearance fitting; retain at least three training views and one holdout for the initial pilot, with usable angular diversity.
5. Select date-compatible independent validation evidence, or identify verified unchanged check areas and record that limitation. An older DSM must not be treated as contemporary ground truth by default.
6. Record a source manifest with capture dates, hashes, access/derivative rights, camera metadata and the recency decision. Only then proceed to full SfM, long Stage 1 training or larger coverage for a current-city model.

This is the revised planning gate. Automated maximum-age enforcement is not implemented yet.

### Historical source and preparation

Public bucket (no credentials, `--no-sign-request`):

```
s3://spacenet-dataset/Hosted-Datasets/CORE3D-Public-Data/Satellite-Images/
  Jacksonville/WV3   289 files, ~133 GB (26 PAN P1BS NITF + MSI)
  Omaha/WV3          473 files
  UCSD/WV2, Richmond/WV2, Tampa/WV2
```

The implemented historical pilot is a 250 m square with a 20 m halo. Eight selected PAN/MSI sources were downloaded and hashed after the remote NITF window-read test proved too slow; local crops preserve RPC offsets. RGB inputs and initial ENU cameras are prepared for three training views and one holdout. These results validate preprocessing, not present-day coverage or a completed reconstruction. See [SatelliteSfM handoff](../sfm/README.md). Do not expand this historical collection for a current-city model.

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
Approved 2015 PAN/MSI + RPCs (versioned source/epoch manifest)
 → Preflight: overlap, angles, clouds, dates          ── fail → imagery/terrain only
 → AOI crop (RPC-aware) + pan-sharpen
 → SatelliteSfM → cameras, sparse points, enu_observer_latlonalt.json
 → Skyfall Stage 1 per block (halo) → fused PLY + normalization record
 → georeference.json + date-compatible independent validation report
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
| 1 | Historical benchmark: convert TerraScope's 12 registered scenes to 3D Tiles; view on the Cesium globe | 1–2 weeks |
| 2 | Complete the approved 2015 one-block pipeline with its ENU anchor preserved, measured Stage 1 training, export and independent validation | 2–4 weeks; runtime and validation prerequisites remain |
| 3 | All blocks of the 1 km² AOI; no seams or doubled opacity between tiles or LOD levels | 2–3 weeks |
| 4 | Semantic voxel tiles + query API | 2–3 weeks |
| 5 | Job runner (resume/retry), versioned releases, catalogue UI | 3+ weeks |

Estimates are planning guesses, not measurements. Stack and layout follow PLAN.md §8.

## 10. Next tasks

1. Resolve and verify the SatelliteSfM runtime, then complete matching, bundle adjustment and sparse reconstruction on the prepared 2015 training views. Preserve the ENU observer and pixel transforms; retain the May 2 holdout.
2. Verify independent validation data for the pilot's capture epoch, coverage and vertical datum. Document any unchanged-area or time-gap limitation.
3. Run measured one-block Skyfall-GS Stage 1 training within the 6 GB GPU budget, preserving normalization and resumable checkpoints.
4. Fuse/export the result, verify geographic placement and quality, then view the historical tiles in CesiumJS before scale-up.

Already completed: Phase 0 rendering checks, isolated CUDA setup and bounded training diagnostic, historical PAN/MSI crops, RGB preparation and RPC-derived initial cameras. Full historical SfM and full training remain unexecuted. The prepared Skyfall JAX_068 diagnostic is separate from this raw-imagery block and does not replace its geographic reconstruction checks.

## Sources

- [CORE3D open repository](https://github.com/pubgeo/core3d-open)
- [US3D / DFC2019 paper](https://arxiv.org/pdf/1811.08739)
- [Urban Semantic 3D dataset](https://ieee-dataport.org/open-access/urban-semantic-3d-dataset)
- [CesiumJS GaussianSplatSorter.js](https://app.unpkg.com/@cesium/engine@26.1.0/files/Source/Scene/GaussianSplatSorter.js)
- [Cesium releases, April 2026](https://cesium.com/blog/2026/04/01/cesium-releases-in-april-2026)

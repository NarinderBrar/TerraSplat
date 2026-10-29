# TerraSplat build plan

Research date: 2026-10-28; status updated 2026-10-29: the 2015 pilot reaches Cesium tiles. Authors' 17-view data gives 2.36 m registered lidar MAE with pilot settings, 1.38 m with their recipe; holdout PSNR falls 21.35 → 18.90 dB. Assumed anchor: shape only. [Comparison](../training/RESULT_2015.md#authors-jax_068-data).

The design below is an engineering proposal. Current upstream capabilities and local evidence are documented separately in [RESEARCH.md](RESEARCH.md). Thresholds are proposed acceptance targets, not measured results.

## 1. Product and first scope

Latest imagery decision (2026-10-29): complete the approved April–May 2015 CORE3D pilot as an end-to-end historical demonstration. Recent source selection is deferred. A future current-city release requires separately qualified recent imagery and date-compatible validation. [PLAN_V2.md](PLAN_V2.md) records the active reconstruction sequence and later freshness gate.

Build a system that accepts a geographic area and an imagery collection, reconstructs a scene, preserves its geographic registration, creates a hierarchy of splat tiles, and publishes a versioned dataset that a browser streams directly. Add basic semantic layers, then language search, once reconstruction and streaming pass validation.

Use the prepared raw 2015 Jacksonville block for the active reconstruction milestone. Existing JAX/NYC scenes support diagnostics and regression where their provenance is verifiable; the local download manifest lists twelve scenes. Do not assume those scenes are contiguous or combine their camera frames without registration evidence.

Active inputs: the approved 2015 CORE3D pilot, with three training acquisitions and a May 2 holdout. Prepared Skyfall-GS JAX/NYC datasets remain separate diagnostic inputs. Current-city imagery is deferred and unselected. Working assumptions:

- Local research data validates the pipeline. The current-city dataset must pass freshness, overlap, high-resolution sensor/RPC, rights and date-compatible validation checks before long reconstruction training.
- Geographic accuracy takes priority over generated visual detail.
- Output and the viewer can be self-hosted. Cesium ion is an optional comparison/managed processing path.
- Start with buildings, vegetation, roads, water, and unknown. Small vehicles and rooftop equipment require a resolution and accuracy study before becoming supported classes.
- GPU hardware and production coverage are unspecified; measure one-scene cost before selecting infrastructure or promising throughput.

## 2. Input feasibility gate

| Input | Planned behavior |
| --- | --- |
| Overlapping stereo/multi-view sensor imagery with RPCs | Preferred satellite reconstruction route; inspect view geometry, dates, clouds, GSD, and calibration |
| Prepared Skyfall images, poses, sparse points | Fastest research integration route; recover the original geographic transform separately |
| Georeferenced drone/aerial images with calibrated poses or SfM | Later input adapter; require metric scale and geographic control |
| One orthomosaic or ordinary satellite basemap images | Offer imagery over terrain/DSM; do not report recovered building geometry from a single view |
| Imagery plus independently measured DSM | Offer a surface-based representation; label the geometry source and its resolution |

A georeferenced raster locates pixels on a reference surface; it does not by itself provide all viewing rays and scene depths needed for 3D reconstruction. Multiple screenshots or map zoom levels do not supply independent stereo observations. Generated unseen facades must be labeled synthesized.

Reject or route insufficient inputs to the imagery/terrain path automatically. Return a specific reason such as missing RPCs, poor overlap, inconsistent epochs, or missing geographic control. View count alone is insufficient; record angular diversity and height sensitivity.

## 3. How the pieces connect

| Component | Responsibility | What we must add |
| --- | --- | --- |
| SatelliteSfM / sensor adapter | Calibration, image preparation, sparse geometry, approximate cameras | Durable geographic transform and quality reports |
| Skyfall-GS Stage 1 | Initial satellite Gaussian reconstruction | Reproducible worker, normalized output contract, held-out evaluation |
| Skyfall-GS Stage 2 | Optional generative refinement | Separate product version and explicit synthesized provenance |
| LangSplat-style semantic adapter | Image language evidence and 3D relevance features | Camera/checkpoint conversion, domain evaluation, stable mapping to exported geometry |
| LOD builder | Spatial partitioning and coarse-to-fine representations | Boundary handling, simplification quality, semantic aggregation |
| glTF / GLB and SPZ | Runtime splat representation and compression | Exact tested extension/version profile and exporter |
| 3D Tiles | Spatial hierarchy and streaming metadata | Bounds, transforms, geometric errors, release validation |
| CesiumJS | Globe, camera, terrain context, tile streaming and rendering | Product UI and a separately validated semantic display/query path |

The output relationship is `tileset.json → tile.glb → splat attributes / embedded compressed data`. SPZ is a codec/container for splats; it does not create reconstruction, LOD, semantics, or geographic registration.

## 4. Pipeline stages and artifacts

1. **Ingest.** Register the AOI, acquisition epoch, image URIs, hashes, rights/provenance, CRS, vertical datum, RPCs/poses, GSD, masks, and optional DEM/DSM/control points. Preserve original files. Use a STAC-shaped catalog record; keep reconstruction metadata in a versioned job manifest.
2. **Preflight.** Confirm the required acquisition-age window and distinguish historical benchmark inputs from current-city candidates. Check geographic overlap and calibration, remove invalid/cloud-covered observations, identify stable acquisition groups, and choose whole-view holdouts before fitting. Record crop/resampling transforms so RPCs and intrinsics remain correct.
3. **Calibrate and initialize.** Run an adapter for prepared data or SatelliteSfM. Produce cameras, sparse geometry or depth prior, calibration residuals, and the complete coordinate conversion chain. Check local perspective/affine approximation error against RPC projection; split blocks or evaluate RPC-native training if it is excessive.
4. **Plan training blocks.** For the pilot use a whole small scene. For larger AOIs choose metric blocks with overlapping halos. Select all cameras that see each block, rather than splitting image pixels into independent XYZ map tiles. Preserve a common geographic frame and shared calibration.
5. **Train reconstruction.** Run Stage 1 in a pinned GPU environment. Save resumable checkpoints, source hashes, settings, losses, and view/geometry quality metrics. Keep generated refinement disabled in the accuracy baseline.
6. **Normalize export.** Produce renderer-ready Gaussian attributes, including any required appearance/filter fusion. Skyfall explicitly requires fused PLY for standalone visualization; a raw training PLY is not sufficient by assumption.
7. **Fit semantic evidence.** Use registered source observations with an occlusion-aware adapter. First project/fuse a small class vocabulary; then evaluate LangSplat-style latent features and open-vocabulary queries. Freeze geometry initially. Never assume a Skyfall checkpoint can be loaded directly by LangSplat.
8. **Build runtime hierarchy.** Partition the canonical model into spatial leaves and generate actual coarse parent approximations. Retain source-to-LOD lineage. Record bounds, geometric errors, tile transforms, counts, hashes, and semantic references.
9. **Package and validate.** Encode the tested GLB/SPZ profile and generate `tileset.json`. Compare decoded attributes, rendered views, orientation, scale, placement, seams, and LOD behavior. Core glTF validation alone does not prove splat-extension correctness.
10. **Publish.** Upload immutable assets into a staging version; verify every reference and checksum; switch the catalog's active version only after validation. Keep rollback references and allow source/epoch-specific rebuilds.

Every stage consumes an immutable manifest and writes a new artifact manifest. Job keys combine input hashes, configuration, model/dependency revisions, and stage version. Retries reuse completed matching artifacts; cancellation preserves checkpoints. Record logs and error causes independently of worker lifetime.

## 5. Geographic registration contract

Automatic placement means preserving known calibration and coordinate transformations, with quality checks. A splat model cannot recover a trustworthy latitude, longitude, and altitude from appearance alone.

Store a `georeference.json` alongside every canonical scene with:

- Horizontal CRS, axis order, units, vertical datum, coordinate epoch when relevant, and source of control.
- The original sensor-to-reconstruction registration and any bundle-adjustment corrections.
- Each centering, rotation, scaling, height shift, and camera convention conversion, including its inverse.
- A local metric frame and its relation to geographic/ECEF coordinates; explicit extent and valid domain for projected CRS conversions.
- Control/checkpoint residuals, uncertainty, validation status, and provenance hashes.

Prefer a local Cartesian frame for splat storage and a double-precision geographic placement transform. A projected UTM coordinate conversion to ECEF is nonlinear: convert properly or establish a justified local tangent approximation; do not treat an arbitrary UTM-to-ECEF conversion as one globally valid rigid matrix. Use smaller anchors for wide areas.

Transform the whole Gaussian, not only its center: for local linear map `A`, use `mean' = A mean + t` and `covariance' = A covariance Aᵀ`. Apply corresponding orientation/scale conversions, and rotate view-dependent spherical harmonics when changing their basis. Keep storage axes, glTF axes, and Cesium placement explicit to avoid double rotations. Convert orthometric and ellipsoidal heights using the appropriate model when required.

Local inspection found conditional Skyfall normalization involving `R`, `T`, radius scaling, and a z shift. Capture those values when executed. The inspected JAX_068 camera JSON has camera/intrinsic fields but no declared geographic CRS or world-placement transform; geographic provenance remains an early gate. If it cannot be recovered, publish that scene only as an explicitly unregistered preview or select a dataset with control.

Validate with independent geographic checkpoints and terrain/DSM comparisons. A visually convincing overlay alone does not establish accuracy. Proposed initial gate: export round-trip numerical error below 0.1 GSD and geographic checkpoint RMSE within a separately agreed source-dependent tolerance. Report horizontal and vertical errors separately.

## 6. Tile and LOD design

Use **separate training and delivery partitions**. Training blocks optimize reconstruction and GPU memory; delivery tiles optimize requests, screen-space quality, and browser memory. One training block can produce many runtime tiles.

For predominantly surface-based satellite scenes, begin with an explicit adaptive quadtree and three-dimensional tile bounds. An octree is a later option for deep vertical scenes. Start with explicit JSON because it is easier to inspect; move to implicit tiling when scale justifies it and the target runtime is verified.

Unlike fixed 2D imagery tiles, splat tiles have volumetric extent, view-dependent appearance, and overlapping kernels. Use adaptive payload budgets rather than a fixed image resolution. Keep XYZ imagery tiling as an independent fallback/base layer.

Proposed first parameters for benchmarking: 256–512 m training cores with 10–20% halos, 50k–200k splats per runtime leaf, and three LOD levels. These are starting experiments; GSD, scene density, kernel size, and GPU memory determine final values. Some existing scenes may be smaller than these defaults.

- Train with halo context; assign final splat ownership deterministically to avoid double opacity across block borders. Check surfaces and kernel footprints across seams, not only centers.
- Account for finite effective Gaussian support in bounds, documenting the cutoff used. Avoid culling splats whose centers are inside one tile but whose visible footprint crosses its boundary.
- Use `REPLACE` refinement as the first hypothesis: parent content approximates its descendants. Confirm actual parent/child transition behavior in the chosen Cesium build. Blindly adding every LOD duplicates opacity.
- Prototype opacity/appearance-aware clustering or simplification. Evaluate render error from representative distances; random point decimation is only a comparison baseline.
- Derive errors in meters from approximation quality, with conservative parent values. Record how view-based error measurements inform them. Tile size alone is not a meaningful LOD quality metric.
- Maintain lineage through pruning, reordering, compression, and parent merges. Parent semantic features use visibility/contribution-aware aggregation; retain unknown or mixed results instead of inventing a dominant class.
- Compare adjacent training blocks and mixed LOD neighbors for gaps, density changes, brightness shifts, and popping.

Run a small third-party tiler evaluation before writing a full implementation. Candidate `CZ-CHO/tiler` advertises adaptive LOD and compressed GLB output, but its README references a different upstream repository that did not resolve during research. Adoption requires inspecting actual code, dependency versions, license, provenance, and output.

## 7. Semantics and interaction

Treat these as separate outputs:

- **Class predictions:** defined ontology, per-class scores, unknown policy, source coverage, and evaluation against annotations.
- **Language relevance:** similarity to a prompt, with model/version and query provenance. Similarity is not a calibrated probability or an object boundary.
- **Reviewed corrections:** explicit annotations stored separately from learned evidence.

Use identical geographic anchors for visual tiles and semantic assets. Define canonical Gaussian IDs by scene version and original index, plus a lineage map for each LOD. Validate mappings after export; array positions in a PLY are not durable IDs. Existing local semantic exports are model-local volumetric fields, so they need a spatial-field adapter, not an assumed per-splat attribute reader.

Keep full embeddings and decoder/checkpoint data on the server initially. Deliver compact class/score data or sparse query results only when requested. An MVP can show precomputed class-colored tilesets sharing the visual hierarchy, or registered vector/point/field overlays. Benchmark the extra storage and memory; do not load every class layer simultaneously.

Current inspected Cesium splat rendering code skips the pick pass and does not establish a supported LangSplat feature-shader API. Therefore interactive semantic picking is a separate implementation task. Begin with backend ray/region queries against a spatial semantic index and show registered result overlays. Test occlusion/depth behavior explicitly. Any custom renderer integration stays behind an adapter.

Evaluate fixed classes on independent scenes/regions and source views excluded from training/correction. Report per-class IoU/F1 where labels permit, coverage, unknown rate, and error by GSD. Evaluate text retrieval separately. The existing local semantic refinement diagnostics used correction inputs, including earlier holdouts; create a new independent test split.

## 8. App, backend, and storage

Proposed stack:

| Layer | Initial choice | Purpose |
| --- | --- | --- |
| Web app | TypeScript, React, Vite, pinned CesiumJS | Globe and project UI; no server rendering requirement |
| API | Python FastAPI | Dataset catalog, jobs, releases, semantic queries |
| Processing | Python CLI stages in isolated containers | Geospatial preparation, training, semantic work |
| Packaging | Existing Node/C++ tool or small adapter | Tested glTF/SPZ/3D Tiles export |
| Metadata | PostgreSQL; PostGIS when geographic querying is implemented | Dataset/job/release metadata and spatial index |
| Artifacts | Local filesystem first, S3-compatible object storage later | Images, checkpoints, immutable tiles and manifests |
| Scheduling | Single durable worker first; queue-backed GPU workers later | Resumption and bounded concurrency |

Keep incompatible CUDA/PyTorch environments separate; exchange files through contracts. Do not merge research repos into one Python environment. A later gsplat-based backend may simplify production dependency choices, but porting satellite cameras, losses, and features remains engineering work.

Viewer flow: select a dataset/AOI → inspect imagery suitability and geographic status → run or resume a job → inspect reconstruction/accuracy evidence → open a release → toggle visual or class layer → query → inspect confidence and provenance. Optional generated versions carry a visible label.

First app features: dataset list, extent navigation, 3D viewing, source imagery/terrain context, layer visibility, release/epoch selection, job progress, quality report, and LOD diagnostics. Start desktop-first; low-memory devices can use imagery/terrain fallback. Use supported terrain/imagery providers with explicit configuration.

Proposed API resources: `datasets`, `jobs`, `releases`, `quality-reports`, and `semantic-queries`. The viewer fetches tile objects directly; the API does not proxy every GLB. Configure CORS, content types, cache headers, authorization for private releases, and range support where required by the selected tools. Cache immutable versioned assets and keep the active-release pointer short-lived.

Suggested implementation layout:

```text
apps/web/
services/api/
pipeline/{ingest,calibration,reconstruction,semantics,tiling,validation}/
adapters/{skyfall,langsplat,exporter}/
schemas/
configs/
deploy/
docs/
```

## 9. Delivery sequence and gates

| Phase | Concrete deliverable | Exit gate |
| --- | --- | --- |
| 0. Evidence and profile | One-scene inventory; original registration; exact exporter, Cesium, extension and SPZ versions | Recover placement or explicitly mark unregistered; tiny fixture and one real splat load correctly |
| 1. Streaming slice | Existing fused model → small multi-level tileset → local static viewer | Correct scale/orientation/placement; network traces prove LOD requests; no duplicate opacity |
| 2. Reproducible pipeline | One job from the approved 2015 inputs through SfM, Stage 1, fusion, tiling and quality report | Resume after interruption; rerun reuses matching stages; holdout and coordinate checks pass; output labelled historical |
| 3. Spatial scale | Two adjacent blocks from a continuous source AOI | Boundary and mixed-LOD checks pass; browser memory stays bounded |
| 4. Semantic pilot | Buildings/vegetation/roads/water/unknown; compact display assets and query API | Independent annotation results reported; unknown supported; mapping survives all LODs |
| 5. Product workflow | Catalog, jobs, releases, epoch selection, staged publication and rollback | End-to-end restart/retry and private/static delivery checks pass |
| 6. Research upgrades | RPC-native alternative, language search, optional generated exploration | Each beats a measured baseline or solves a documented product need |

Indicative effort: phases 0–1, roughly 1–2 engineer-weeks if a usable model and geographic provenance are available; phases 2–3, another 3–5; phases 4–5, another 3–6. These are planning estimates, not commitments. Missing source registration, new satellite camera support, and custom semantic rendering can materially extend them. Avoid setting a city-scale deadline until the initial benchmark is complete.

Record per scene: calibration wall time, GPU hours, peak VRAM, training/splat counts, image/GSD statistics, source and output bytes, quality metrics, first-visible-content time, request bytes, p50/p95 frame times, decode time, and browser memory. Provisional browser target: 30 FPS during a fixed desktop camera route and first visible content within 5 s at a declared network/device profile. Document hardware and warm/cold cache conditions.

Estimate cost using measured GPU-hours × selected rate, plus preprocessing, storage, and egress. For area projections, account for overlap, acquisition count, and scene complexity; do not extrapolate one tile's runtime as a guaranteed linear city-scale rate. Compressed bytes do not predict decoded GPU memory.

## 10. Decisions still requiring project input

Choose and verify a recent imagery source, acceptable capture age, within-group acquisition span, date-compatible validation evidence and access/derivative rights; target AOI size and GSD; available GPU/VRAM; horizontal/vertical accuracy requirement; hosting and access model; and whether generated detail is a desired product mode. Resolve these when needed by the phase gates. No purchases, external uploads, or deployments are part of this planning work.

Next implementation priority: product web app v1 reading pipeline releases. The fixed-weight MoGe experiment regressed registered lidar MAE from 2.82 to 7.83 m and holdout PSNR from 17.73 to 15.18 dB; retain the five-view baseline ([depth supervision](../training/RESULT_2015.md#depth-supervision)). Historical training, fusion, tiling and Cesium delivery are complete. Broader geographic validation and recent imagery qualification remain later work.

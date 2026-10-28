# Research and compatibility notes

Checked 2026-10-28. Web pages and repository main branches are moving references, not a pinned release manifest. The implementation must record exact commits and package versions. Sources describe upstream behavior; none of the proposed toolchain combinations were executed during planning.

## Sources supplied by the user

- [Cesium LOD announcement](https://cesium.com/blog/2026/04/27/3d-gaussian-splats-lod/): demonstrates the overall glTF-payload/3D-Tiles-hierarchy approach and describes managed photo reconstruction in ion. Its photo workflow does not establish support for our satellite RPC inputs. It describes 3D Tiles 2.0 as proposed; do not claim this plan requires a finalized 2.0 specification.
- [OGC announcement](https://www.ogc.org/blog-article/ogc-khronos-and-geospatial-leaders-add-3d-gaussian-splats-to-the-gltf-asset-standard/): ecosystem and interoperability context; use the actual extension and renderer implementation for compatibility decisions.
- [CesiumJS LOD tutorial](https://cesium.com/learn/cesiumjs-learn/3d-guassian-splat-tilesets-lods/): useful viewer baseline and screen-space-error/inspector workflow. A hosted demonstration does not establish our custom exporter compatibility.
- [CesiumJS repository](https://github.com/CesiumGS/cesium): chosen globe/runtime candidate.
- [glTF repository](https://github.com/KhronosGroup/glTF): format specifications, not a trainer or tile generator.
- [SPZ repository](https://github.com/nianticlabs/spz): compression implementation, not a spatial index.

## Critical format findings

The current [base `KHR_gaussian_splatting` specification](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_gaussian_splatting/README.md) explicitly reports ratified status. It describes splat attributes on point primitives and extensibility. This status does not prove every renderer implements the same revision.

[Cesium's migration issue](https://github.com/CesiumGS/cesium/issues/12837) documents the replacement of `KHR_spz_gaussian_splats_compression` with `KHR_gaussian_splatting` plus `KHR_gaussian_splatting_compression_spz_2`. The [changelog](https://github.com/CesiumGS/cesium/blob/main/CHANGES.md) records removal of the older extension and subsequent transform, memory, and sorting fixes. Pin and test a release containing the fixes relevant to our data.

The [SPZ compression proposal](https://github.com/KhronosGroup/glTF/pull/2531) is separate from the ratified base extension. The expected compression README was absent from the Khronos main directory during inspection. Treat the exact codec profile as a compatibility gate, rather than declaring all SPZ-in-glTF combinations standardized.

The [current SPZ README](https://github.com/nianticlabs/spz/blob/main/README.md) describes version 4 as the writer default and continued reading of earlier versions. It also documents axis conventions and conversions. A recent library's default output must not be assumed compatible with a renderer's SPZ-v2 extension path. Record the file-header version and verify the actual decoder. Compression gains vary with the model and do not indicate GPU memory savings.

Use a compatibility report with these fields:

| Field | Status before implementation |
| --- | --- |
| CesiumJS exact release/commit | Not selected |
| Exporter exact revision and dependency lock | Not selected |
| Base extension schema/profile | Ratified source available; exporter/runtime match untested |
| Compression extension identifier | Cesium documents `_compression_spz_2`; selected runtime must confirm |
| Embedded SPZ file version | Must be explicitly selected and decoded successfully |
| SH degree, kernel/filter, color space, axes | Must match fused source and target renderer |
| Per-splat metadata, picking, custom shaders | Not assumed supported |
| Structural and rendered validation | Pending |

## Local reference evidence

These paths were inspected read-only. They contain local additions; upstream documentation alone does not describe all available work.

| Local evidence | Implication |
| --- | --- |
| `Ref/Skyfall-GS/data/download-manifest.json` | Twelve JAX/NYC scenes listed, pinned dataset revision `82bcd8ae044afcf3182d335a47d0694ba2b4daa2`, about 2.34 GB listed bytes |
| `Ref/Skyfall-GS/README.md` | Stage 1 reconstruction and Stage 2 iterative generative synthesis are separate; visualization requires fused PLY |
| `Ref/Skyfall-GS/scene/dataset_readers.py`, `readSatelliteInfo` | Conditional rotation/translation, radius scaling, and z translation must survive any export |
| `Ref/Skyfall-GS/data/datasets_JAX/JAX_068/transforms_train.json` | Inspected top-level keys are `camera_model`, `frames`, `h`, `w`; no declared world CRS/placement contract |
| `Ref/LangSplat/scripts/prepare_skyfall_pilot.py` | Existing off-center-camera rectification adapter preserves model frame; rejects extra `R`/`T` normalization without an explicit adapter |
| `Ref/LangSplat/scripts/export_semantic_categories.py` | Existing class-field extraction operates on registered evidence; useful adapter reference |
| `Ref/LangSplat/output/semantic_categories_v1/JAX_068/JAX_068-building-glow.json` | Candidate output is a model-local 512×512×64 scalar field, 16 MiB for this category; not a standard glTF semantic payload |
| `Ref/LangSplat/validation/semantic_refinement_v2/README.md` | Some corrections use manually reviewed stadium boundaries and earlier holdout views; source diagnostics are not independent validation |

Reuse camera conventions, manifest patterns, and verified artifacts after inspecting the corresponding code. Preserve manual-correction provenance. Existing checkpoints and fields were not re-rendered or re-evaluated for this plan.

The local Skyfall top-level license is Apache-2.0 but it also includes an Inria Gaussian-Splatting license file. LangSplat's local license and rasterizer license contain research-use terms. This is a dependency-selection flag: inspect the actual code paths, model weights, datasets, and licenses before choosing a production backend. It is not evidence that the entire combined pipeline has one license. A file-based adapter does not change dependency terms.

## Additional dots to evaluate

| Candidate | Why it matters | Planned treatment |
| --- | --- | --- |
| [Skyfall-specific SatelliteSfM fork](https://github.com/jayin92/SatelliteSfM) | Explicit preprocessing route for RPC satellite images and COLMAP reconstructions | First preprocessing reference; capture all calibration/normalization metadata |
| [EOGS paper, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Aira_Gaussian_Splatting_for_Efficient_Satellite_Image_Photogrammetry_CVPR_2025_paper.pdf) | Satellite-specific reconstruction using non-orthorectified observations and camera metadata | Geometry baseline alongside Skyfall Stage 1 |
| [RPC-GS, June 2026](https://arxiv.org/abs/2606.06690) | Direct RPC projection rather than approximate perspective/affine training cameras | Research candidate if camera approximation error is material; availability/export quality not validated |
| [SatSplat, June 2026](https://arxiv.org/abs/2606.28581) | 2DGS surface reconstruction with camera adjustment and illumination handling | Compare DSM quality; its surface representation needs a separate compatibility/export study |
| [SatSplat project](https://gdaosu.github.io/satsplat/) | Author benchmark and code link | Reported numbers are dataset/hardware-specific, not our performance forecast |
| [gsplat](https://github.com/nerfstudio-project/gsplat) and [license](https://github.com/nerfstudio-project/gsplat/blob/main/LICENSE) | Apache-2.0 rasterization/training foundation | Candidate for a future backend; not a drop-in satellite pipeline or license clearance for other dependencies |
| [CZ-CHO/tiler](https://github.com/CZ-CHO/tiler) | Advertises splat PLY → adaptive LOD → SPZ/GLB 3D Tiles and geographic placement | Small-fixture evaluation only; referenced upstream URL did not resolve during research |
| [3D Tiles specification](https://github.com/CesiumGS/3d-tiles/blob/main/specification/README.adoc) | Hierarchy, transforms, bounds, content, and metadata contracts | Reference for the delivery manifest; keep experimental splat profile explicit |

RPC-GS and SatSplat are additional research directions, not dependencies required to start the project. Evaluate reconstruction accuracy independently from browser export compatibility.

## Renderer-specific semantic limitation

The inspected [Cesium `GaussianSplatPrimitive` source](https://github.com/CesiumGS/cesium/blob/main/packages/engine/Source/Scene/GaussianSplatPrimitive.js) returns early during the pick pass. This finding does not prove every possible interaction is unsupported, but it rules out assuming native per-Gaussian feature picking. No LangSplat decoder/custom-feature integration was established by that source inspection. The plan therefore uses a backend spatial query adapter and separate result overlays initially.

## Questions the first experiments must answer

1. Can a local scene's exact metric and geographic registration be recovered from trustworthy source metadata?
2. Does a tiny anisotropic colored fixture survive export with the correct axes, scale, opacity, quaternion order, SH basis, and file version?
3. Does one actual fused scene load in the pinned Cesium build, and do camera movements request distinct LOD content?
4. Can adjacent and mixed-LOD tiles avoid opacity seams and holes?
5. How much do semantic fields cost in decoded browser memory, and can sparse/tiled alternatives replace whole-scene atlases?
6. What accuracy remains on annotations and views never used for training, refinement, threshold tuning, or manual correction?

Do not infer affirmative answers from screenshots, standard announcements, or a successful single-file splat viewer.

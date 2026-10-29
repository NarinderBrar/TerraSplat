# TerraSplat

Pipeline for turning suitable satellite imagery into georeferenced, tiled Gaussian splats. Phase 0 includes a static Cesium diagnostic viewer.

More views: 19 training acquisitions plus May 2 holdout; basic/recipe PSNR 16.64/15.49 dB. Both exports fail the existing extent gate, so lidar comparison is blocked and the five-view release stays active. Next: Phase 3: two neighbouring blocks (shared ownership grid). [Result](training/RESULT_2015.md#nineteen-training-views).

```bash
npm ci
npm run build
npm test
npm run test:browser
npm run dev
```

See [Phase 0 instructions and compatibility contract](docs/PHASE0.md) and
[Phase 2 input validation and remaining prerequisites](docs/PHASE2.md).
For the isolated GPU setup and bounded diagnostic, see [Stage 1 training](training/README.md).
For the completed measured run and failure history, see [the Stage 1 result](training/RESULT_2015.md).
For the prepared 2015 block and loader checks, see [the pilot training handoff](training/PILOT.md).
For raw sensor crops and RPC preservation, see [Imagery preparation](imagery/README.md).
For the historical multi-view inputs and initial ENU cameras, see [SatelliteSfM handoff](sfm/README.md).

- [Web app](apps/web/README.md): globe, release footprints and quality panels; run `npm run dev:app`.
- [Releases and read-only API](docs/RELEASES.md): immutable local publication, activation and viewer contracts.
- [Plan v2](docs/PLAN_V2.md): selected 2015 pilot, reconstruction sequence, CesiumJS and the 6 GB GPU budget.
- [Build plan](docs/PLAN.md): architecture, data contracts, implementation order, acceptance gates, and open decisions.
- [Training findings](docs/TRAINING_FINDINGS.md): what improved heights across six runs, and the recommended setup for product training (about 15–20 views, authors' recipe).
- [Research and compatibility notes](docs/RESEARCH.md): sources, local reference findings, and candidates to evaluate.
- [Editable architecture diagram](docs/architecture.drawio): open in draw.io/diagrams.net. The desktop exporter was unavailable in the planning environment.

The trained 2015 model is fused and exported with its placement transforms verified for consistency (see [the result](training/RESULT_2015.md#fused-export)). It is tiled as SPZ 3D Tiles and renders in CesiumJS: open `pilot.html` with `npm run dev` after generating tiles (see [the tiles result](training/RESULT_2015.md#3d-tiles-and-cesium)). An independent lidar check puts it 1.8 m horizontally and within about 0.2–0.6 m vertically of the DFC2019 lidar, with 3.3 m height MAE from a rendered-depth DSM ([details](training/RESULT_2015.md#independent-lidar-check)). Training on five views instead of three cut that to 2.8 m (buildings 3.7 → 2.9 m; [details](training/RESULT_2015.md#five-training-views)). Static MoGe depth supervision regressed MAE to 7.83 m and holdout PSNR to 15.18 dB ([details](training/RESULT_2015.md#depth-supervision)); retain the five-view baseline. The product web app v1 reads pipeline releases. Broader independent validation remains required. Recent imagery selection is deferred.
# TerraSplat

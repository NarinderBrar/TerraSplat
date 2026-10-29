# TerraSplat

Pipeline for turning suitable satellite imagery into georeferenced, tiled Gaussian splats. Phase 0 includes a static Cesium diagnostic viewer.

Phase 0 compatibility checks pass. Phase 2 has completed CPU sparse reconstruction, skew correction and a verified Skyfall input handoff for the approved April–May 2015 Jacksonville pilot. Measured Stage 1 training is next. Full splat reconstruction, tiling at scale, and the product web app are not implemented.

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
For the prepared 2015 block and loader checks, see [the pilot training handoff](training/PILOT.md).
For raw sensor crops and RPC preservation, see [Imagery preparation](imagery/README.md).
For the historical multi-view inputs and initial ENU cameras, see [SatelliteSfM handoff](sfm/README.md).

- [Plan v2](docs/PLAN_V2.md): selected 2015 pilot, reconstruction sequence, CesiumJS and the 6 GB GPU budget.
- [Build plan](docs/PLAN.md): architecture, data contracts, implementation order, acceptance gates, and open decisions.
- [Research and compatibility notes](docs/RESEARCH.md): sources, local reference findings, and candidates to evaluate.
- [Editable architecture diagram](docs/architecture.drawio): open in draw.io/diagrams.net. The desktop exporter was unavailable in the planning environment.

Next milestone: run measured Skyfall Stage 1 training on the prepared 2015 block, retain the holdout and placement transforms, and export a historical demonstration. Recent imagery selection is deferred.
# TerraSplat

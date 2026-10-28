# TerraSplat

Pipeline for turning suitable satellite imagery into georeferenced, tiled Gaussian splats. Phase 0 includes a static Cesium diagnostic viewer.

Phase 0 implementation is underway. Reconstruction, tiling at scale, and the product web app are not implemented.

```bash
npm ci
npm run build
npm test
npm run test:browser
npm run dev
```

See [Phase 0 instructions and compatibility contract](docs/PHASE0.md).

- [Plan v2](docs/PLAN_V2.md): agreed prototype (Jacksonville CORE3D imagery, CesiumJS, 6 GB GPU budget, TerraScope reuse).
- [Build plan](docs/PLAN.md): architecture, data contracts, implementation order, acceptance gates, and open decisions.
- [Research and compatibility notes](docs/RESEARCH.md): sources, local reference findings, and candidates to evaluate.
- [Editable architecture diagram](docs/architecture.drawio): open in draw.io/diagrams.net. The desktop exporter was unavailable in the planning environment.

Recommended first milestone: one existing scene → verified coordinate transform → small LOD tileset → self-hosted CesiumJS viewer. Resolve input provenance and exporter compatibility before automating large training jobs.
# TerraSplat

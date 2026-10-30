# Releases

The local publish stage turns a completed tiled run into an immutable release. It requires only stdlib + numpy;
use the existing `.venv-sfm` (Python 3.12). Configs in `configs/releases/` record dataset metadata,
train/holdout capture times from the selections, source runs and JSON pointers into committed quality evidence.

```bash
.venv-sfm/bin/python -m pipeline.publish --config configs/releases/jax068-2015-3v.json --releases artifacts/releases
.venv-sfm/bin/python -m pipeline.publish --config configs/releases/jax068-2015-5v.json --releases artifacts/releases --activate
```

```text
artifacts/releases/
  catalog.json
  index/jacksonville/tileset.json
  index/new-york/tileset.json
  <release_id>/{release.json,tileset.json,tile-plan.json,tiles-report.json,tiles/,evidence/}
```

The pipeline dataset `jax068-2015` keeps one release, `jax068-2015-19v-r1` (active, 21 tiles, registered lidar
MAE 1.36 m). The earlier 3-view and 5-view releases (`jax068-2015-3v-r1/r2`, `-5v-r1/r2`) were removed from the
catalogue and disk on 2026-10-30; their measurements remain in `training/` and their training runs are kept.
Skyfall datasets are listed in [SKYFALL_TILES.md](SKYFALL_TILES.md).

The publisher verifies plan, tileset and GLB hashes against `tiles-report.json`, checks tile references,
copies delivery files and evidence into `.staging/<release_id>/`, validates and rehashes, then renames atomically.
Training payload binaries are not delivery assets and are not copied. A release ID can never be overwritten;
rebuilds need a new ID. Failed prepublication attempts remove their staging folder. A process killed before rename
may leave staging behind; inspect and remove that incomplete staging folder before retrying.

`release.json` includes UTC creation time, the pipeline Git commit, source/config hashes, capture times,
ENU anchor, owned crop GeoJSON polygon (longitude/latitude), six-value geographic bbox with ellipsoidal height,
quality, evidence and limitations. Footprints transform the crop square at ENU up=0 through ECEF to WGS84;
bbox heights cover the owned centre Z range, including tangent-plane curvature. They describe ownership,
not kernel support. Full kernel bounds remain in the tileset. Pipeline releases are historical, cover one block,
have no ground control and use lidar of unknown acquisition date; they are not production accuracy claims.
`bytes` counts delivery files and evidence excluding `release.json`; `quality.glb_bytes` counts GLBs alone.
`quality.splats` counts owned crop splats rather than duplicate LOD splats.

The manifest hashes every other release file. The catalogue hashes the manifest itself, avoiding a self-hash cycle.
JSON Schemas live in `schemas/`; publication validates their used assertion subset with a fail-closed stdlib validator,
and tests also validate documents and schemas with the full Draft 2020-12 `jsonschema` implementation.

Activation and rollback only switch the catalogue pointer; every listed release remains available:

```bash
.venv-sfm/bin/python -m pipeline.publish --releases artifacts/releases --activate-release skyfall-jax-068-stage1-r1
.venv-sfm/bin/python -m pipeline.publish --releases artifacts/releases --activate-release skyfall-jax-068-final-r2
```

Publish/activation uses a filesystem lock. Catalogue and index writes each use a temporary file and atomic replacement;
the catalogue is authoritative. A crash between replacements can leave the index stale; repair with
`python -m pipeline.publish --releases artifacts/releases --rebuild-index`. A crash after release rename but before
catalogue replacement can leave a complete unlisted release: preserve it, inspect its manifest/hashes, then recover
the catalogue from a backup or publish under a new ID. Multi-file transactional publication is not provided.

Each regional 3D Tiles 1.1 index references only active releases as external tilesets with relative URIs.
Child ECEF boxes contain all transformed corners of the release root box, including kernel support;
the external root applies its own ENU transform exactly once. Child geometric error matches the referenced
tileset's error of drawing nothing; the index root exceeds its children's errors and spans their bounds.
`ADD` at the content-free index root supports independent blocks; release hierarchies retain `REPLACE`.

An optional `ground_points` config entry (a rendered-depth `depth-points.npz`) adds `ground` to the manifest: the
median of the modal 0.5 m bin of the model's own 1 m DSM inside the owned crop, as an ellipsoidal height. It comes
from the model, not the lidar. The viewer uses it to place base imagery at the model ground: terrain is the release
ground within 2 km of each active footprint and eases back to the ellipsoid by 8 km. Without terrain, imagery sits
on the ellipsoid about 20 m above the Jacksonville ground and would hide the splats. Releases without `ground`
fall back to fading the globe near the surface. `jax068-2015-*-r2` add this field; the `-r1` releases remain for rollback.

The [read-only API](../services/api/README.md) exposes dataset/release lists, quality and active footprints
(including `ground_height_m`).
Part 2 adds `apps/web/` (React/Vite/Cesium), loading the regional index and offering dataset/release selection,
3D Footprints, quality, navigation and LOD diagnostics. The existing pilot viewer remains separate.

Publisher tests run in `tests/sfm/`; API tests run separately in `tests/api/` with `.venv-api`.
The shared `tests/conftest.py` fixture is opt-in, imports publisher dependencies lazily and defines no collection hooks.

Skyfall Stage 1 releases use the authors' prepared data and TerraScope visual registrations;
see [Skyfall tiles](SKYFALL_TILES.md). Missing independent lidar measurements are null, never zero.
Pipeline release quality must remain measured.

One-block orchestration, offline CPU preparation, resume, stage reuse and isolated
job publication are documented in [JOBS.md](JOBS.md).

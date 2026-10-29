# Release API

Read-only FastAPI service for the React/Vite/Cesium app. Run from the repository root:

```bash
uv venv --python 3.12 .venv-api
uv pip install --python .venv-api/bin/python -r services/api/requirements.txt
.venv-api/bin/python -m uvicorn services.api.main:app --host 127.0.0.1 --port 8000
```

`TERRASPLAT_RELEASES_DIR` defaults to `artifacts/releases` relative to the repository root.
An override can be absolute or relative to the process working directory.
`TERRASPLAT_CORS_ORIGINS` is a comma-separated allowlist; defaults are
`http://127.0.0.1:5173,http://127.0.0.1:4173`. Tile requests also receive CORS headers.
No jobs, authentication, database, uploads or mutation endpoints are provided.

| GET endpoint | Response |
| --- | --- |
| `/api/health` | `{"status":"ok"}` (liveness) |
| `/api/datasets` | Dataset list with active release summary, quality and tile URL |
| `/api/datasets/{dataset_id}` | Dataset and all releases for comparison/rollback selection |
| `/api/releases/{release_id}` | Complete release manifest |
| `/api/releases/{release_id}/quality` | Quality summary, limitations, evidence URLs and hashes |
| `/api/footprints` | GeoJSON FeatureCollection of active crop footprints; capture range, headline MAE and tile URL |
| `/api/index/{region}` | `tileset_url` for the active regional index |

IDs must match `[a-z0-9-]+` (422 otherwise); unknown IDs return 404. URLs are origin-relative.
OpenAPI is at `/openapi.json`, interactive docs at `/docs`.
Quality distances are metres, PSNR is dB, completeness is a fraction, and splats counts owned crop splats.

Cesium loads `/releases/index/{region}/tileset.json` directly, then streams external release tilesets and GLBs.
`StaticFiles` handles file delivery, conditional GET and HTTP ranges; `.glb` is `model/gltf-binary`,
`.json` is `application/json`. Release files use `Cache-Control: public, max-age=31536000, immutable`;
`catalog.json` and regional indexes use `public, max-age=60`. Staging, lock files, unlisted files,
directory listings and traversal are inaccessible. Keep the filesystem writable only by the publisher.
The metadata endpoints read current files on each request. The index may remain cached for 60 seconds after activation.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH= .venv-api/bin/python -m pytest -q tests/api
```

Dependencies, including test tools and transitive versions, are pinned in `requirements.txt`.
The installed Starlette version emits an HTTPX deprecation warning during tests; HTTPX TestClient remains functional.
See [Releases](../../docs/RELEASES.md) for publishing and rollback.

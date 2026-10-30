"""Filesystem-backed release metadata and direct static tile delivery."""

import json
import mimetypes
import os
from pathlib import Path
import re
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from services.api.models import (
    Dataset, DatasetSummary, FeatureCollection, Health, IndexLink, JobStatus, Manifest, QualityReport, ReleaseSummary,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ORIGINS = 'http://127.0.0.1:5173,http://127.0.0.1:4173'
mimetypes.add_type('model/gltf-binary', '.glb')
mimetypes.add_type('application/json', '.json')


def valid_id(value):
    if not re.fullmatch('[a-z0-9-]+', value):
        raise HTTPException(422, 'Identifiers must match [a-z0-9-]+')
    return value


class ReleaseFiles(StaticFiles):
    async def get_response(self, path, scope):
        parts = Path(path).parts
        # Never expose staging, lock files or unlisted directories.
        if not parts or any(p.startswith('.') for p in parts):
            raise HTTPException(404, 'File not found')
        mutable = path == 'catalog.json' or (len(parts) == 3 and parts[0] == 'index'
                                             and re.fullmatch('[a-z0-9-]+', parts[1])
                                             and parts[2] == 'tileset.json')
        if not mutable:
            if not re.fullmatch('[a-z0-9-]+', parts[0]) or len(parts) < 2:
                raise HTTPException(404, 'File not found')
            manifest_path = Path(self.directory) / parts[0] / 'release.json'
            if not manifest_path.is_file():
                raise HTTPException(404, 'File not found')
            manifest = json.loads(manifest_path.read_text())
            relative = Path(*parts[1:]).as_posix()
            if relative != 'release.json' and relative not in manifest['files']:
                raise HTTPException(404, 'File not found')
        response = await super().get_response(path, scope)
        if response.status_code in (200, 206, 304):
            response.headers['Cache-Control'] = (
                'public, max-age=60' if mutable else 'public, max-age=31536000, immutable')
        return response


def job_status(name, state, now=None):
    """Summary of a pipeline.job state file: progress and the running or failed stage, without local paths."""
    now = now or datetime.now(timezone.utc)
    stages = state.get('stages', {})
    done = sum(s.get('status') == 'done' for s in stages.values())
    current = next(((n, s) for n, s in stages.items() if s.get('status') in ('running', 'failed')), None)
    times = [t for s in stages.values() for t in (s.get('start_time'), s.get('end_time')) if t]
    if current:
        name_, stage = current
        started = datetime.fromisoformat(stage['start_time']) if stage.get('start_time') else None
        ended = datetime.fromisoformat(stage['end_time']) if stage.get('end_time') else now
        elapsed = (ended - started).total_seconds() if started else None
        current = {'name': name_, 'status': stage['status'], 'gpu': bool(stage.get('gpu')), 'elapsed_seconds': elapsed}
    # A job stopped before a GPU stage (--no-gpu) has no running stage and pending GPU stages.
    pending_gpu = any(s.get('status') != 'done' and s.get('gpu') for s in stages.values())
    status = (current['status'] if current else 'done' if done == len(stages) and stages
              else 'stopped' if done else 'pending')
    if status == 'stopped' and not pending_gpu:
        status = 'pending'
    return {'name': name, 'stages_total': len(stages), 'stages_done': done, 'state': status, 'current': current,
            'updated_at': max(times) if times else None}


def create_app(releases_dir=None, allowed_origins=None, jobs_dir=None):
    directory = Path(releases_dir or os.environ.get('TERRASPLAT_RELEASES_DIR', ROOT / 'artifacts/releases')).resolve()
    jobs = Path(jobs_dir or os.environ.get('TERRASPLAT_JOBS_DIR', ROOT / 'artifacts/jobs')).resolve()
    origins = allowed_origins if allowed_origins is not None else [
        origin.strip() for origin in os.environ.get('TERRASPLAT_CORS_ORIGINS', DEFAULT_ORIGINS).split(',')
        if origin.strip()]
    app = FastAPI(title='TerraSplat release API', version='1.0.0')
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=['GET', 'HEAD'],
                       allow_headers=['Range'], expose_headers=['Accept-Ranges', 'Content-Range', 'ETag'])

    def load(path):
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            raise HTTPException(404, 'Resource not found') from None

    def catalogue():
        path = directory / 'catalog.json'
        return load(path)['datasets'] if path.is_file() else []

    def manifest(release_id):
        valid_id(release_id)
        if not any(r['release_id'] == release_id for d in catalogue() for r in d['releases']):
            raise HTTPException(404, 'Release not found')
        return load(directory / release_id / 'release.json')

    def summary(release_id):
        m = manifest(release_id)
        return ReleaseSummary(**{k: m[k] for k in ('release_id', 'title', 'capture_dates', 'quality', 'bytes')},
                              manifest_url=f'/api/releases/{release_id}',
                              tileset_url=f'/releases/{release_id}/{m["tileset_path"]}')

    def dataset_summary(d):
        return {k: d[k] for k in ('dataset_id', 'title', 'region', 'active_release_id')} | {
            'active_release': summary(d['active_release_id']) if d['active_release_id'] else None}

    @app.get('/api/health')
    def health() -> Health:
        return Health()

    @app.get('/api/datasets')
    def datasets() -> list[DatasetSummary]:
        return [dataset_summary(d) for d in catalogue()]

    @app.get('/api/datasets/{dataset_id}')
    def dataset(dataset_id: str) -> Dataset:
        valid_id(dataset_id)
        for d in catalogue():
            if d['dataset_id'] == dataset_id:
                return dataset_summary(d) | {'releases': [summary(r['release_id']) for r in d['releases']]}
        raise HTTPException(404, 'Dataset not found')

    # Omit absent optional fields while preserving explicit nulls for unmeasured quality.
    @app.get('/api/releases/{release_id}', response_model_exclude_unset=True)
    def release(release_id: str) -> Manifest:
        return manifest(release_id)

    @app.get('/api/releases/{release_id}/quality')
    def quality(release_id: str) -> QualityReport:
        m = manifest(release_id)
        return {'release_id': release_id, 'quality': m['quality'], 'limitations': m['limitations'],
                'evidence': [{'name': name, 'url': f'/releases/{release_id}/{name}', 'sha256': m['files'][name]}
                             for name in m['evidence']]}

    @app.get('/api/footprints')
    def footprints() -> FeatureCollection:
        features = []
        for d in catalogue():
            if not d['active_release_id']:
                continue
            m = manifest(d['active_release_id'])
            dates = m['capture_dates']['train'] + m['capture_dates']['holdout']
            features.append({'type': 'Feature', 'geometry': m['footprint'], 'properties': {
                'dataset_id': m['dataset_id'], 'release_id': m['release_id'], 'title': m['title'],
                'capture_date_start': min(dates), 'capture_date_end': max(dates),
                'headline_mae_m': m['quality']['registered_lidar_mae_all_m'],
                'tileset_url': f'/releases/{m["release_id"]}/{m["tileset_path"]}',
                'source_label': m.get('source_label', 'TerraSplat raw-imagery pipeline'),
                'accuracy_label': m.get('accuracy_label', 'Known anchor; no ground control'),
                'ground_height_m': (m.get('ground') or {}).get('ellipsoidal_height_m')}})
        return {'type': 'FeatureCollection', 'features': features}

    @app.get('/api/index/{region}')
    def index(region: str) -> IndexLink:
        valid_id(region)
        if (not any(d['region'] == region for d in catalogue())
                or not (directory / 'index' / region / 'tileset.json').is_file()):
            raise HTTPException(404, 'Region not found')
        return {'region': region, 'tileset_url': f'/releases/index/{region}/tileset.json'}

    @app.get('/api/jobs')
    def job_list() -> list[JobStatus]:
        """Pipeline jobs under the jobs directory (one folder per job with job-state.json)."""
        result = []
        for path in sorted(jobs.glob('*/job-state.json')) if jobs.is_dir() else []:
            try:
                result.append(job_status(path.parent.name, json.loads(path.read_text())))
            except (OSError, ValueError, KeyError):
                continue  # a state file being rewritten; it is atomic, so skip until the next poll
        return result

    app.mount('/releases', ReleaseFiles(directory=directory, check_dir=False), name='releases')
    return app


app = create_app()

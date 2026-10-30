"""Publish verified tiled runs; stdlib + numpy (use .venv-sfm)."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIMITATIONS = (
    "Historical demonstration of one block. Lidar acquisition date unknown; scene changes may affect metrics. "
    "No ground control; independent geographic accuracy remains unverified. Not production quality. "
    "Needle-like splats, border streaks and loose kernel bounds remain; view-dependent colour is baked."
)


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch('[a-z0-9-]+', value):
        raise ValueError(f'Invalid identifier: {value!r}')
    return value


def relative_file(directory, name):
    path = directory / name
    if Path(name).is_absolute() or '..' in Path(name).parts or not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError(f'Unsafe relative path: {name}')
    if not path.is_file() or path.is_symlink():
        raise ValueError(f'Missing or symlink file: {name}')
    return path


def validate(value, schema, location='$'):
    """Validate the assertion subset used by our schemas, without a runtime dependency.

    Full Draft 2020-12 validation and schema checks also run with jsonschema in tests.
    Fail closed if a schema gains an unsupported keyword.
    """
    supported = {'$schema', '$id', 'title', 'description', 'type', 'properties', 'required',
                 'additionalProperties', 'items', 'minItems', 'maxItems', 'pattern', 'const', 'minimum', 'maximum'}
    if set(schema) - supported:
        raise ValueError(f'Unsupported schema keywords: {set(schema) - supported}')
    kinds = {'object': dict, 'array': list, 'string': str, 'number': (int, float),
             'integer': int, 'boolean': bool, 'null': type(None)}
    kind = schema.get('type')
    choices = kind if isinstance(kind, list) else [kind]
    if kind and not any(isinstance(value, kinds[k]) and
                        (k not in ('number', 'integer') or not isinstance(value, bool)) for k in choices):
        raise ValueError(f'{location}: expected {kind}')
    if 'const' in schema and value != schema['const']:
        raise ValueError(f'{location}: unexpected value')
    if isinstance(value, dict):
        if set(schema.get('required', [])) - value.keys():
            raise ValueError(f'{location}: missing required fields')
        properties = schema.get('properties', {})
        for key, item in value.items():
            child = properties.get(key, schema.get('additionalProperties', {}))
            if child is False:
                raise ValueError(f'{location}: unexpected field {key}')
            if isinstance(child, dict):
                validate(item, child, f'{location}.{key}')
    if isinstance(value, list):
        if len(value) < schema.get('minItems', 0) or len(value) > schema.get('maxItems', math.inf):
            raise ValueError(f'{location}: invalid length')
        for item in value:
            validate(item, schema.get('items', {}), location + '[]')
    if isinstance(value, str) and 'pattern' in schema and not re.search(schema['pattern'], value):
        raise ValueError(f'{location}: invalid string')
    if isinstance(value, (int, float)):
        if (not math.isfinite(value) or value < schema.get('minimum', -math.inf)
                or value > schema.get('maximum', math.inf)):
            raise ValueError(f'{location}: invalid number')


def validate_document(value, name):
    validate(value, read(ROOT / 'schemas' / f'{name}.schema.json'))


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.', dir=path.parent)
    try:
        with open(fd, 'w') as stream:
            json.dump(value, stream, separators=(',', ':'), allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        Path(name).replace(path)
    finally:
        Path(name).unlink(missing_ok=True)


@contextmanager
def publication_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.publish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def ecef_to_geodetic(x, y, z):
    """WGS84 ECEF metres -> longitude/latitude degrees, ellipsoidal height metres."""
    a, e2 = 6378137.0, 6.6943799901413165e-3
    p = math.hypot(x, y)
    if p < 1e-9:
        if abs(z) < 1e-9:
            raise ValueError('ECEF origin has no geodetic coordinate')
        return 0.0, math.copysign(90.0, z), abs(z) - a * math.sqrt(1 - e2)
    lat = math.atan2(z, p * (1 - e2))
    for _ in range(15):
        n = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)
        updated = math.atan2(z + e2 * n * math.sin(lat), p)
        if abs(updated - lat) < 1e-15:
            lat = updated
            break
        lat = updated
    # Dot with the ellipsoid normal; stable at equator and poles.
    height = p * math.cos(lat) + z * math.sin(lat) - a * math.sqrt(1 - e2 * math.sin(lat) ** 2)
    return math.degrees(math.atan2(y, x)), math.degrees(lat), height


def footprint(plan):
    transform = np.array(plan['root_transform']).reshape(4, 4, order='F')
    half = plan['crop']['half_extent_m']
    square = [(-half, -half), (half, -half), (half, half), (-half, half), (-half, -half)]
    polygon = plan['crop'].get('polygon_enu_m', [[e,n,0] for e,n in square])
    ring = [list(ecef_to_geodetic(*(transform @ [*p, 1])[:3])[:2]) for p in polygon]
    heights = [ecef_to_geodetic(*(transform @ [e, n, z, 1])[:3])[2]
               for e, n, _ in polygon[:4] for z in plan['crop']['owned_z_range_m']]
    # The minimum curvature contribution can occur in the interior at the anchor.
    heights += [ecef_to_geodetic(*(transform @ [0, 0, z, 1])[:3])[2]
                for z in plan['crop']['owned_z_range_m']]
    return {'type': 'Polygon', 'coordinates': [ring]}, [
        min(p[0] for p in ring), min(p[1] for p in ring), max(p[0] for p in ring), max(p[1] for p in ring),
        min(heights), max(heights)]


def ground_height(points_path, plan, cell_m=1.0, bin_m=0.5, surface_name='rendered-depth DSM'):
    """Display ground: the most common height of the model's own rendered-depth DSM inside the owned crop.

    Points are ENU metres about the plan's anchor. The value comes from the model, not the lidar, so the
    viewer can place base imagery at the model's ground instead of the ellipsoid.
    """
    enu = np.load(points_path)['enu']
    half = plan['crop']['half_extent_m']
    if 'ownership_grid' in plan['crop']:
        from pipeline.ownership import owned_mask
        keep = owned_mask(enu, {'enu_to_ecef': np.array(plan['root_transform']).reshape(4,4,order='F')},
                          plan['crop']['ownership_grid'], plan['crop']['ownership_cell'])
    else:
        keep = (np.abs(enu[:, 0]) <= half) & (np.abs(enu[:, 1]) <= half)
    enu = enu[keep]
    if len(enu) == 0:
        raise ValueError('No ground points inside the owned crop')
    cells = np.floor((enu[:, :2] + half) / cell_m).astype(np.int64)
    key = cells[:, 0] * (int(2 * half / cell_m) + 2) + cells[:, 1]
    order = np.lexsort((enu[:, 2], key))
    surface = enu[order, 2][np.r_[key[order][1:] != key[order][:-1], True]]
    edges = np.arange(np.floor(surface.min()), np.ceil(surface.max()) + bin_m, bin_m)
    counts, edges = np.histogram(surface, bins=edges)
    low = edges[counts.argmax()]
    mode = float(np.median(surface[(surface >= low) & (surface < low + bin_m)]))
    return {'ellipsoidal_height_m': plan['enu_anchor']['ellipsoidal_height_m'] + mode,
            'method': (f'median of the modal {bin_m:g} m bin of the {surface_name} ({cell_m:g} m cells) '
                       'inside the owned crop'),
            'source_sha256': sha256(points_path)}


def catalog(directory):
    path = directory / 'catalog.json'
    return read(path) if path.exists() else {'schema_version': 1, 'datasets': []}


def build_indexes(directory, document):
    regions = {d['region'] for d in document['datasets']}
    for region in regions:
        children, corners, declared = [], [], {}
        for dataset in document['datasets']:
            if dataset['region'] != region or dataset['active_release_id'] is None:
                continue
            release_id = identifier(dataset['active_release_id'])
            tileset = read(directory / release_id / 'tileset.json')
            root = tileset['root']
            transform = np.array(root['transform']).reshape(4, 4, order='F')
            box = np.array(root['boundingVolume']['box'])
            axes = box[3:].reshape(3, 3)
            points = [(transform @ [*(box[:3] + np.array(sign) @ axes), 1])[:3]
                      for sign in itertools.product([-1, 1], repeat=3)]
            low, high = np.min(points, axis=0), np.max(points, axis=0)
            centre, half = (low + high) / 2, (high - low) / 2
            volume = {'box': [*centre, half[0], 0, 0, 0, half[1], 0, 0, 0, half[2]]}
            corners.extend(points)
            for key in ('extensionsUsed', 'extensionsRequired'):
                declared.setdefault(key, set()).update(tileset.get(key, []))
                content = tileset.get('extensions', {}).get('3DTILES_content_gltf', {})
                declared.setdefault('gltf_' + key, set()).update(content.get(key, []))
            # An external-content tile has the referenced tileset's geometricError.
            # Do not copy its transform here: the external root applies it once.
            children.append({'boundingVolume': volume, 'geometricError': tileset['geometricError'],
                             'content': {'uri': f'../../{release_id}/tileset.json'}})
        if corners:
            low, high = np.min(corners, axis=0), np.max(corners, axis=0)
            centre, half = (low + high) / 2, (high - low) / 2
            error = max(float(np.linalg.norm(high - low)), max(c['geometricError'] for c in children))
            error = math.nextafter(error, math.inf)
            volume = {'box': [*centre, half[0], 0, 0, 0, half[1], 0, 0, 0, half[2]]}
        else:
            volume, error = {'sphere': [0, 0, 0, 1]}, 1
        index = {'asset': {'version': '1.1'}, 'geometricError': error,
                 'root': {'boundingVolume': volume, 'geometricError': error, 'refine': 'ADD', 'children': children}}
        # Viewers such as Cesium pick content decoders from the outermost tileset, so repeat the releases'
        # declarations here.
        for key in ('extensionsUsed', 'extensionsRequired'):
            if declared.get(key):
                index[key] = sorted(declared[key])
        gltf = {key: sorted(declared['gltf_' + key]) for key in ('extensionsUsed', 'extensionsRequired')
                if declared.get('gltf_' + key)}
        if gltf:
            index['extensions'] = {'3DTILES_content_gltf': gltf}
        atomic_json(directory / 'index' / region / 'tileset.json', index)


def save_catalog(directory, document):
    validate_document(document, 'catalog')
    # Each file is atomic; catalog is authoritative if interrupted between replacements.
    atomic_json(directory / 'catalog.json', document)
    build_indexes(directory, document)


def activate(directory, release_id):
    directory = Path(directory)
    identifier(release_id)
    with publication_lock(directory):
        document = catalog(directory)
        for dataset in document['datasets']:
            if any(r['release_id'] == release_id for r in dataset['releases']):
                entry = next(r for r in dataset['releases'] if r['release_id'] == release_id)
                if sha256(directory / release_id / 'release.json') != entry['manifest_sha256']:
                    raise ValueError('Manifest hash mismatch')
                manifest = read(directory / release_id / 'release.json')
                verify_release(directory / release_id, manifest)
                dataset['active_release_id'] = release_id
                save_catalog(directory, document)
                return
        raise ValueError(f'Unknown release: {release_id}')


def verify_release(directory, manifest):
    validate_document(manifest, 'release-manifest')
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
    if actual != set(manifest['files']) | {'release.json'}:
        raise ValueError('Release file inventory mismatch')
    for name, expected in manifest['files'].items():
        if sha256(relative_file(directory, name)) != expected:
            raise ValueError(f'Hash mismatch: {name}')


def publish(config_path, directory, make_active=False, recover_identical=False):
    config_path, directory = Path(config_path), Path(directory)
    config = read(config_path)
    for key in ('dataset_id', 'release_id', 'region'):
        identifier(config[key])
    with publication_lock(directory):
        release_id = config['release_id']
        destination = directory / release_id
        if destination.exists():
            manifest = read(destination / 'release.json') if recover_identical else None
            if manifest is None or manifest['source']['config_sha256'] != sha256(config_path):
                raise ValueError(f'Immutable release already exists: {release_id}')
            verify_release(destination, manifest)
            source = ROOT / config['tiles_dir']
            for name, key in [('tile-plan.json', 'tile_plan_sha256'),
                              ('tiles-report.json', 'tiles_report_sha256')]:
                if sha256(source / name) != manifest['source'][key]:
                    raise ValueError(f'Publication recovery input changed: {name}')
            for name, expected in manifest['files'].items():
                if name.startswith('tiles/') or name in ('tile-plan.json', 'tileset.json', 'tiles-report.json'):
                    if sha256(relative_file(source, name)) != expected:
                        raise ValueError(f'Publication recovery input changed: {name}')
            document = catalog(directory)
            dataset = next((d for d in document['datasets'] if d['dataset_id'] == config['dataset_id']), None)
            if dataset is None:
                dataset = {'dataset_id': config['dataset_id'], 'title': config['dataset_title'],
                           'region': config['region'], 'active_release_id': None, 'releases': []}
                document['datasets'].append(dataset)
            if dataset['region'] != config['region']:
                raise ValueError('Dataset region cannot change')
            entry = next((r for r in dataset['releases'] if r['release_id'] == release_id), None)
            manifest_hash = sha256(destination / 'release.json')
            if entry is not None and entry['manifest_sha256'] != manifest_hash:
                raise ValueError('Manifest hash mismatch')
            if entry is None:
                dataset['releases'].append({'release_id': release_id, 'title': manifest['title'],
                                           'manifest_path': f'{release_id}/release.json',
                                           'manifest_sha256': manifest_hash})
            if make_active:
                dataset['active_release_id'] = release_id
            save_catalog(directory, document)
            return manifest
        source = ROOT / config['tiles_dir']
        report = read(source / 'tiles-report.json')
        hashes = {'tile-plan.json': report['planSha256'], 'tileset.json': report['tilesetSha256']}
        hashes.update({f'tiles/{identifier(k)}.glb': v['glbSha256'] for k, v in report['tiles'].items()})
        for name, expected in hashes.items():
            if sha256(relative_file(source, name)) != expected:
                raise ValueError(f'Hash mismatch: {name}')
        plan, tileset = read(source / 'tile-plan.json'), read(source / 'tileset.json')
        def references(node):
            found = [node['content']['uri']] if 'content' in node else []
            return found + [uri for child in node.get('children', []) for uri in references(child)]
        if set(references(tileset['root'])) != {n for n in hashes if n.endswith('.glb')}:
            raise ValueError('Tileset references differ from report')
        if {t['id'] for t in plan['tiles']} != set(report['tiles']):
            raise ValueError('Tile plan differs from report')
        staging_name = release_id + '-' + uuid.uuid4().hex if recover_identical else release_id
        staging = directory / '.staging' / staging_name
        staging.mkdir(parents=True, exist_ok=False)
        try:
            for name in [*hashes, 'tiles-report.json']:
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(relative_file(source, name), target)
            evidence = {}
            for name in config['evidence']:
                path = ROOT / name
                target = staging / 'evidence' / path.name
                if target.exists():
                    raise ValueError('Duplicate evidence basename')
                target.parent.mkdir(exist_ok=True)
                shutil.copyfile(path, target)
                evidence[name] = read(target)
            quality = dict(config.get('quality_values', {}))
            for key, ref in config['quality_sources'].items():
                value = evidence[ref['file']]
                for part in ref['pointer'].strip('/').split('/'):
                    value = value[int(part)] if isinstance(value, list) else value[part]
                quality[key] = value
            geometry, bbox = footprint(plan)
            ground = ground_height(ROOT / config['ground_points'], plan,
                                   surface_name=config.get('ground_surface', 'rendered-depth DSM')) \
                if 'ground_points' in config else None
            files = {p.relative_to(staging).as_posix(): sha256(p) for p in sorted(staging.rglob('*')) if p.is_file()}
            for name, expected in hashes.items():
                if files[name] != expected:
                    raise ValueError(f'Hash mismatch after copy: {name}')
            manifest = {
                'schema_version': 1, 'dataset_id': config['dataset_id'], 'release_id': release_id,
                'title': config['title'], 'description': config['description'],
                'created_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                'pipeline_git_commit': subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'source': {'run_directory': config['stage1_run_dir'], 'tiles_directory': config['tiles_dir'],
                           'config_sha256': sha256(config_path), 'tile_plan_sha256': report['planSha256'],
                           'tiles_report_sha256': files['tiles-report.json'],
                           'export_manifest_sha256': plan['export_manifest_sha256'],
                           'fused_ply_sha256': plan['fused_ply_sha256']},
                'capture_dates': config['capture_dates'], 'historical': True, 'anchor': plan['enu_anchor'],
                'footprint': geometry, 'bbox': bbox, 'tileset_path': 'tileset.json',
                'tile_count': len(report['tiles']),
                'bytes': sum(p.stat().st_size for p in staging.rglob('*') if p.is_file()),
                'quality': quality, 'evidence': [f'evidence/{Path(n).name}' for n in config['evidence']],
                'files': files, 'limitations': LIMITATIONS,
            }
            if ground:
                manifest['ground'] = ground
            for key in ('source_label', 'accuracy_label'):
                if key in config:
                    manifest[key] = config[key]
            unavailable = {k for k, v in quality.items() if v is None}
            optional_lidar = {'registered_lidar_mae_all_m', 'registered_lidar_mae_ground_m',
                              'registered_lidar_mae_buildings_m', 'registered_lidar_mae_vegetation_m',
                              'registered_lidar_rmse_m', 'horizontal_offset_m', 'completeness'}
            missing_truth = (config.get('lidar_truth_unavailable') is True
                             and unavailable <= optional_lidar
                             and config.get('accuracy_label') == 'No independent lidar truth supplied')
            if not config['dataset_id'].startswith('skyfall-') and unavailable and not missing_truth:
                raise ValueError('Pipeline release quality must be measured')
            validate_document(manifest, 'release-manifest')
            atomic_json(staging / 'release.json', manifest)
            verify_release(staging, read(staging / 'release.json'))
            document = catalog(directory)
            dataset = next((d for d in document['datasets'] if d['dataset_id'] == config['dataset_id']), None)
            if dataset is None:
                dataset = {'dataset_id': config['dataset_id'], 'title': config['dataset_title'],
                           'region': config['region'],
                           'active_release_id': None, 'releases': []}
                document['datasets'].append(dataset)
            if dataset['region'] != config['region']:
                raise ValueError('Dataset region cannot change')
            dataset['releases'].append({'release_id': release_id, 'title': manifest['title'],
                                        'manifest_path': f'{release_id}/release.json',
                                        'manifest_sha256': sha256(staging / 'release.json')})
            if make_active:
                dataset['active_release_id'] = release_id
            validate_document(document, 'catalog')
            staging.rename(destination)
            save_catalog(directory, document)
            return manifest
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--releases', type=Path, default=Path('artifacts/releases'))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config', type=Path)
    group.add_argument('--activate-release', help='Activate an existing release (also supports rollback)')
    group.add_argument('--rebuild-index', action='store_true', help='Repair indexes from the authoritative catalog')
    parser.add_argument('--activate', action='store_true', help='Activate the newly published release')
    parser.add_argument('--recover-identical', action='store_true', help='Recover a verified identical publication after interruption')
    args = parser.parse_args()
    if args.config:
        result = publish(args.config, args.releases, args.activate, args.recover_identical)
        print(json.dumps({k: result[k] for k in ('release_id', 'tile_count', 'bytes')}))
    elif args.activate_release:
        activate(args.releases, args.activate_release)
    else:
        with publication_lock(args.releases):
            build_indexes(args.releases, catalog(args.releases))


if __name__ == '__main__':
    main()

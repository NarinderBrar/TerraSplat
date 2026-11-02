"""Semantics v5 classes for the 1 km grid blocks (configs/semantics-v5.json method). Run with .venv-imagery.

Each block's class grid covers its owned 250 m cell plus a margin, on the DFC2019 JAX_068 0.5 m lattice. Blocks that
overlap the JAX_068 lidar are scored there (owned cells only) and their measured placement offset is removed, as in v5;
the others get classes without a score. A Skyfall authors' scene gets a 700 m grid around its centre and no score (its
DFC2019 labels are kept for the blind test). Probabilities come from pipeline.landcover_predict (.venv-landcover).

    .venv-imagery/bin/python -m pipeline.semantics_grid block --job artifacts/jobs/grid-1km-a2 \
        --probabilities <landcover_predict output> --output <new dir>
    .venv-imagery/bin/python -m pipeline.semantics_grid summary --blocks <block outputs...> --output summary.json
    .venv-imagery/bin/python -m pipeline.semantics_grid scene --scene artifacts/skyfall/JAX_004/scene \
        --export artifacts/skyfall/JAX_004/final-placed --points <render_depth --half-extent 350 output> --probabilities <dir> --output <new dir>
    .venv-imagery/bin/python -m pipeline.semantics_grid release --job artifacts/jobs/grid-1km-a2 \
        --segments <pipeline.segments output> --output release.json   (--job also takes a release config file)
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from pipeline.dsm_validation import read_truth
from pipeline.phase2 import digest
from pipeline.semantics import CODES, cell_lonlat, evaluate, preview, splat_classes, top_surface
from pipeline.semantics_clip import aligned_georeference, cell_model_points
from pipeline.semantics_landcover import NAMES, check_prediction, classify, fuse_views

TRUTH = 'artifacts/reference/dfc2019-jax068'
MARGIN_M = 20.0
UNSCORED = 65
HALF_EXTENT_M = 350.0  # authors' scenes: their training photos see 97 % of the ground out to ±350 m (JAX_004)


def stages(job):
    state = json.loads((Path(job) / 'job-state.json').read_text())['stages']
    return {name: Path(record['output']) for name, record in state.items() if record['status'] == 'done'}


def block_inputs(job):
    done = stages(job)
    plan = json.loads((done['encoding'] / 'tiles' / 'tile-plan.json').read_text())
    validation = done['dsm_validation'] / 'validation' / 'dsm-validation.json' if 'dsm_validation' in done else None
    return {'scene': done['skyfall_inputs'] / 'scene', 'export': done['export'] / 'export',
            'depth_points': done['render_depth'] / 'depth' / 'depth-points.npz', 'tiles': done['encoding'] / 'tiles',
            'plan': plan, 'registration': validation, 'publish': done['publish']}


def enu_to_lonlat(enu, enu_to_ecef):
    from pyproj import Transformer

    frame = np.asarray(enu_to_ecef, np.float64)
    ecef = enu @ frame[:3, :3].T + frame[:3, 3]
    lon, lat, _ = Transformer.from_crs(4978, 4979, always_xy=True).transform(ecef[:, 0], ecef[:, 1], ecef[:, 2])
    return lon, lat


def lonlat_to_enu(lon, lat, enu_to_ecef):
    from pyproj import Transformer

    frame = np.asarray(enu_to_ecef, np.float64)
    ecef = np.stack(Transformer.from_crs(4979, 4978, always_xy=True).transform(lon, lat, np.zeros_like(lon)), 1)
    return (ecef - frame[:3, 3]) @ frame[:3, :3]


def owned_cell(plan):
    crop = plan['crop']
    grid = crop['ownership_grid']
    return grid['enu_to_ecef'], grid['cells'][crop['ownership_cell']]


def block_grid(plan, truth, margin_m=MARGIN_M):
    """Square grid on the truth lattice covering the owned cell plus a margin (only 'dsm' shape is used)."""
    from pyproj import Transformer

    frame, cell = owned_cell(plan)
    corners = np.array([[e, n, 0.0] for e in cell['east_m'] for n in cell['north_m']])
    east, north = Transformer.from_crs(4326, 32617, always_xy=True).transform(*enu_to_lonlat(corners, frame))
    resolution = truth['resolution']
    west = truth['west'] + math.floor((min(east) - margin_m - truth['west']) / resolution) * resolution
    top = truth['north'] + math.ceil((max(north) + margin_m - truth['north']) / resolution) * resolution
    size = math.ceil(max(max(east) + margin_m - west, top - (min(north) - margin_m)) / resolution)
    return {'dsm': np.zeros((size, size)), 'west': west, 'north': top, 'resolution': resolution}


def owned_mask(grid, plan):
    """Cells whose centre lies in the block's half-open owned cell (shared grid ENU frame)."""
    frame, cell = owned_cell(plan)
    lon, lat = cell_lonlat(grid)
    enu = lonlat_to_enu(np.asarray(lon), np.asarray(lat), frame)
    (e0, e1), (n0, n1) = cell['east_m'], cell['north_m']
    inside = (enu[:, 0] >= e0) & (enu[:, 0] < e1) & (enu[:, 1] >= n0) & (enu[:, 1] < n1)
    return inside.reshape(grid['dsm'].shape)


def truth_on_grid(grid, truth):
    """Truth classes resampled onto the block grid (same lattice); UNSCORED outside the truth tile."""
    resolution = truth['resolution']
    column = round((grid['west'] - truth['west']) / resolution)
    row = round((truth['north'] - grid['north']) / resolution)
    size, limit = grid['dsm'].shape[0], truth['classes'].shape[0]
    result = np.full((size, size), UNSCORED, truth['classes'].dtype)
    r0, r1, c0, c1 = max(0, -row), min(size, limit - row), max(0, -column), min(size, limit - column)
    if r0 < r1 and c0 < c1:
        result[r0:r1, c0:c1] = truth['classes'][r0 + row:r1 + row, c0 + column:c1 + column]
    return result


def run_block(config_path, job, probs_dir, output, truth_dir=TRUTH):
    inputs = block_inputs(job)
    truth = read_truth(truth_dir, 'JAX_068')
    grid = block_grid(inputs['plan'], truth)
    truth_classes = truth_on_grid(grid, truth)
    owned = owned_mask(grid, inputs['plan'])
    if not (owned & (truth_classes != UNSCORED)).any():
        inputs = inputs | {'registration': None}  # only blocks the lidar measured
    return run_grid(config_path, inputs, grid, owned, truth_classes, probs_dir, output,
                    {'job': str(job), 'cell': inputs['plan']['crop']['ownership_cell']})


def scene_grid(export_dir, half_extent_m=HALF_EXTENT_M, resolution=0.5):
    """Square grid centred on the scene's ENU origin, as pipeline.render_depth's nadir render."""
    from pyproj import Transformer

    geo = json.loads((Path(export_dir) / 'export-manifest.json').read_text())['georeference']
    lon, lat = enu_to_lonlat(np.zeros((1, 3)), geo['enu_to_ecef'])
    east, north = Transformer.from_crs(4326, 32617, always_xy=True).transform(lon[0], lat[0])
    size = math.ceil(2 * half_extent_m / resolution)
    return {'dsm': np.zeros((size, size)), 'resolution': resolution,
            'west': math.floor((east - half_extent_m) / resolution) * resolution,
            'north': math.floor((north + half_extent_m) / resolution) * resolution}


def run_scene(config_path, scene, export_dir, depth_points, probs_dir, output, half_extent_m=HALF_EXTENT_M):
    """Classes for a Skyfall authors' scene; no truth is read (its DFC2019 labels are kept for the blind test)."""
    rendered = json.loads((Path(depth_points).parent / 'depth-report.json').read_text())['camera']['half_extent_m']
    if rendered < half_extent_m:
        raise ValueError(f'Depth was rendered for ±{rendered} m; render with --half-extent {half_extent_m}')
    grid = scene_grid(export_dir, half_extent_m)
    shape = grid['dsm'].shape
    inputs = {'scene': Path(scene), 'export': Path(export_dir), 'depth_points': Path(depth_points), 'registration': None}
    return run_grid(config_path, inputs, grid, np.ones(shape, bool), np.full(shape, UNSCORED, np.uint8), probs_dir,
                    output, {'scene': str(scene), 'cell': Path(export_dir).parent.name})


def run_grid(config_path, inputs, grid, owned, truth_classes, probs_dir, output, source):
    import rasterio

    config = json.loads(Path(config_path).read_text())
    output, probs_dir = Path(output), Path(probs_dir)
    if output.exists():
        raise ValueError('Use a new output directory')
    prediction = check_prediction(probs_dir, config)
    scored = owned & (truth_classes != UNSCORED) & (truth_classes != 17)
    registration = inputs['registration']

    manifest = json.loads((inputs['export'] / 'export-manifest.json').read_text())
    geo, shift = aligned_georeference(manifest['georeference'], registration)
    manifest = manifest | {'georeference': geo}
    shape = grid['dsm'].shape
    top, valid, filled, ground, hag = top_surface(inputs['depth_points'], geo, grid, shift)
    points = cell_model_points(grid, filled, geo)
    tolerance = 1.0 / np.asarray(geo['model_to_enu'])[0][0]  # 1.0 m in model units
    probabilities, views, views_used = fuse_views(inputs['scene'], probs_dir, points, tolerance)
    probabilities, views = probabilities.reshape(shape + (len(NAMES),)), views.reshape(shape)
    classes, model_only, label, known = classify(probabilities, views, hag, valid, config['classes']['raised_hag_m'],
                                                 config['scoring']['min_views'])
    splat_class, inside, surface, wall = splat_classes(manifest, grid, top, ground, classes, classes == CODES['building'])

    mapping = config['evaluation']['mapping']
    scores = None
    if scored.any():
        scores = {'cells': int(scored.sum()), 'evaluation': evaluate(classes, truth_classes, mapping, scored),
                  'model_only_diagnostic': evaluate(model_only, truth_classes, mapping, scored)}

    output.mkdir(parents=True)
    profile = {'driver': 'GTiff', 'width': shape[1], 'height': shape[0], 'count': 1, 'crs': 'EPSG:32617',
               'transform': rasterio.transform.from_origin(grid['west'], grid['north'], grid['resolution'], grid['resolution'])}
    for filename, array, dtype in (('class-raster.tif', classes, 'uint8'), ('model-only-raster.tif', model_only, 'uint8'),
                                   ('views.tif', views, 'uint8'), ('hag.tif', hag, 'float32'),
                                   ('owned.tif', owned, 'uint8'), ('truth.tif', truth_classes, 'uint8')):
        with rasterio.open(output / filename, 'w', dtype=dtype, **profile) as sink:
            sink.write(array.astype(dtype), 1)
    np.savez_compressed(output / 'cell-probabilities.npz', probabilities=probabilities.astype(np.float16), names=NAMES)
    np.savez_compressed(output / 'splat-classes.npz', classes=splat_class, legend=json.dumps(CODES))
    preview(output / 'preview.png', np.where(owned, classes, 0), truth_classes, scored, mapping)
    counts = lambda array: {name: int((array == code).sum()) for name, code in CODES.items()}
    report = {
        'schema_version': 1, 'config': str(config_path), 'config_sha256': digest(Path(config_path)),
        **source,
        'export_manifest_sha256': digest(inputs['export'] / 'export-manifest.json'),
        'depth_points_sha256': digest(inputs['depth_points']), 'weights_sha256': prediction['weights_sha256'],
        'grid': {'west': grid['west'], 'north': grid['north'], 'resolution': grid['resolution'], 'size': shape[0],
                 'owned_cells': int(owned.sum())},
        'views': views_used, 'registration': str(registration) if registration else None, 'enu_shift_m': list(shift),
        'cells': {'owned_with_evidence': int((known & owned).sum()), 'owned_by_class': counts(np.where(owned, classes, 255)),
                  'model_label': {n: int((known & owned & (label == n)).sum()) for n in NAMES}},
        'splats': {'total': int(len(splat_class)), 'inside_grid': int(inside.sum()), 'surface': int(surface.sum()),
                   'walls': int(wall.sum()), 'by_class': counts(splat_class)},
        'scores': scores,
        'notes': ['Classes use the v5 rules unchanged; scores cover owned cells inside the DFC2019 JAX_068 lidar only.'],
    }
    (output / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(output / 'scored-cells.npz', classes=classes[scored], model_only=model_only[scored],
                        truth=truth_classes[scored])
    return report


def summary(config_path, blocks, output):
    """One score over the owned, lidar-covered cells of all blocks (owned cells do not overlap)."""
    config = json.loads(Path(config_path).read_text())
    mapping, targets = config['evaluation']['mapping'], config['evaluation']['targets']
    parts = [np.load(Path(block) / 'scored-cells.npz') for block in blocks]
    parts = [part for part in parts if len(part['truth'])]
    truth = np.concatenate([part['truth'] for part in parts])
    every = np.ones(len(truth), bool)
    result = {name: evaluate(np.concatenate([part[name] for part in parts]), truth, mapping, every)
              for name in ('classes', 'model_only')}
    e = result['classes']
    checks = {'overall_accuracy': e['overall_accuracy'] >= targets['overall_accuracy_min'],
              'building_iou': (e['building']['iou'] or 0) >= targets['building_iou_min'],
              'vegetation_iou': (e['vegetation']['iou'] or 0) >= targets['vegetation_iou_min']}
    reports = [json.loads((Path(block) / 'evaluation.json').read_text()) for block in blocks]
    document = {'schema_version': 1, 'config': str(config_path), 'scored_cells': int(len(truth)),
                'scored_blocks': [r['cell'] for r in reports if r['scores']],
                'evaluation': e, 'model_only_diagnostic': result['model_only'],
                'targets': targets, 'checks': checks, 'meets_targets': all(checks.values()),
                'blocks': {r['cell']: (r['scores'] and {'cells': r['scores']['cells'],
                                                        'overall_accuracy': r['scores']['evaluation']['overall_accuracy'],
                                                        'building_iou': r['scores']['evaluation']['building']['iou'],
                                                        'vegetation_iou': r['scores']['evaluation']['vegetation']['iou']})
                           for r in reports}}
    Path(output).write_text(json.dumps(document, indent=2) + '\n')
    return document


def release_config(base, segments_dir, output, release_suffix='r2'):
    """A published release config (a grid job, or a release config file) with a segment layer and a new release id."""
    path = Path(base)
    config = json.loads((path if path.is_file() else block_inputs(path)['publish'] / 'release.json').read_text())
    previous = config['release_id']
    layer = json.loads((Path(segments_dir) / 'segments.json').read_text())
    config.update(release_id=f"{previous.rsplit('-r', 1)[0]}-{release_suffix}", segments_dir=str(Path(segments_dir).absolute()),
                  description=config['description'] + f" {release_suffix} adds a segment layer ({layer['label']}); "
                  f'tiles are identical to {previous}.')
    Path(output).write_text(json.dumps(config, indent=2) + '\n')
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    block = commands.add_parser('block')
    block.add_argument('--config', default='configs/semantics-v5.json')
    block.add_argument('--job', required=True)
    block.add_argument('--probabilities', required=True)
    block.add_argument('--truth', default=TRUTH)
    block.add_argument('--output', required=True)
    total = commands.add_parser('summary')
    total.add_argument('--config', default='configs/semantics-v5.json')
    total.add_argument('--blocks', nargs='+', required=True)
    total.add_argument('--output', required=True)
    scene = commands.add_parser('scene')
    scene.add_argument('--config', default='configs/semantics-v5.json')
    scene.add_argument('--scene', required=True, help='Prepared Skyfall scene (masks/, transforms_train.json)')
    scene.add_argument('--export', required=True)
    scene.add_argument('--points', required=True, help='pipeline.render_depth depth-points.npz')
    scene.add_argument('--probabilities', required=True)
    scene.add_argument('--half-extent', type=float, default=HALF_EXTENT_M)
    scene.add_argument('--output', required=True)
    release = commands.add_parser('release')
    release.add_argument('--job', required=True, help='Grid job directory, or a release config file')
    release.add_argument('--segments', required=True)
    release.add_argument('--suffix', default='r2', help='New release revision')
    release.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'block':
        report = run_block(args.config, args.job, args.probabilities, args.output, args.truth)
        s = report['scores']
        print(f"{report['cell']}: {report['splats']['by_class']}" + (
            f"; scored {s['cells']} cells, accuracy {s['evaluation']['overall_accuracy']:.3f}, building IoU "
            f"{s['evaluation']['building']['iou']:.3f}, vegetation IoU {s['evaluation']['vegetation']['iou']:.3f}" if s else ''))
    elif args.command == 'scene':
        report = run_scene(args.config, args.scene, args.export, args.points, args.probabilities, args.output, args.half_extent)
        print(f"{report['cell']}: {report['splats']['by_class']}")
    elif args.command == 'summary':
        print(json.dumps(summary(args.config, args.blocks, args.output), indent=2))
    else:
        print(json.dumps(release_config(args.job, args.segments, args.output, args.suffix), indent=2))


if __name__ == '__main__':
    main()

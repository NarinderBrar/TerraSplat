"""Semantics v5: OpenEarthMap land-cover probabilities fused onto the DFC2019 grid (configs/semantics-v5.json).

Run with .venv-imagery after pipeline.landcover_predict (.venv-landcover) has written <name>_probs.npy for each
training view (see docs/SEMANTICS.md). Projection, visibility, fusion and scoring are v3/v4's (pipeline.semantics_clip).

    .venv-imagery/bin/python -m pipeline.semantics_landcover --config configs/semantics-v5.json \
        --export <export dir> --points <depth-points.npz> --truth <dfc2019 dir> --scene <skyfall scene> \
        --probabilities <landcover_predict output> --registration <dsm-validation.json> --output <new dir>
"""

import argparse
import json
from pathlib import Path
import warnings

import numpy as np

from pipeline.dsm_validation import read_truth
from pipeline.phase2 import digest
from pipeline.semantics import CODES, evaluate, preview, splat_classes, top_surface
from pipeline.semantics_clip import aligned_georeference, cell_model_points, frame_projection, visible_pixels

NAMES = ['bareland', 'rangeland', 'developed space', 'road', 'tree', 'water', 'agriculture land', 'building']
# OpenEarthMap class -> (code when raised, code when flat); v4's rules with OpenEarthMap names
RULES = {
    'tree': ('vegetation', 'vegetation'), 'rangeland': ('grass', 'grass'), 'building': ('building', 'ground'),
    'road': ('unknown', 'road'), 'developed space': ('unknown', 'ground'), 'bareland': ('unknown', 'ground'),
    'agriculture land': ('unknown', 'ground'), 'water': ('unknown', 'water'),
}
MODEL_ONLY = {'tree': 'vegetation', 'rangeland': 'grass', 'building': 'building', 'road': 'road',
              'developed space': 'ground', 'bareland': 'ground', 'agriculture land': 'ground', 'water': 'water'}


def classify(probabilities, views, hag, valid, raised_m, min_views):
    known = valid & (views >= min_views) & np.isfinite(probabilities).all(axis=-1)
    label = np.asarray(NAMES)[np.argmax(np.nan_to_num(probabilities, nan=-1.0), axis=-1)]
    raised = hag >= raised_m
    classes = np.zeros(hag.shape, np.uint8)
    model_only = np.zeros(hag.shape, np.uint8)
    for name, (high, low) in RULES.items():
        here = known & (label == name)
        classes[here] = np.where(raised[here], CODES[high], CODES[low])
        model_only[here] = CODES[MODEL_ONLY[name]]
    return classes, model_only, label, known


def view_probabilities(scene, frame, transforms, probs_dir, points, tolerance):
    from PIL import Image

    name = Path(frame['file_path']).stem
    P, (width, height) = frame_projection(frame, transforms)
    probs = np.load(probs_dir / f'{name}_probs.npy', mmap_mode='r')
    if probs.shape[:2] != (height, width) or probs.shape[2] != len(NAMES):
        raise ValueError(f'{name}: probabilities {probs.shape} do not match the {width} x {height} view')
    projected = np.column_stack((points, np.ones(len(points)))) @ P.T
    depth = projected[:, 2]
    u, v = projected[:, 0] / depth, projected[:, 1] / depth
    coverage = np.asarray(Image.fromarray(np.load(scene / 'masks' / f'{name}.npy')).resize((width, height), Image.NEAREST))
    ok, rows, cols = visible_pixels(u, v, np.where(depth > 0, depth, np.nan), (height, width), coverage, tolerance)
    result = np.full((len(points), len(NAMES)), np.nan, np.float32)
    result[ok] = probs[rows[ok], cols[ok]]
    return result


def check_prediction(probs_dir, config):
    prediction = json.loads((Path(probs_dir) / 'report.json').read_text())
    if prediction['weights_sha256'] != config['model']['weights_sha256']:
        raise ValueError('Probabilities come from different model weights than the config')
    settings = config['prediction']
    if (prediction['classes'] != NAMES or prediction['window'] != settings['window']
            or prediction['overlap'] != settings['overlap'] or prediction['flip_tta'] != settings['flip_tta']):
        raise ValueError('Prediction settings differ from the config')
    return prediction


def fuse_views(scene, probs_dir, points, tolerance):
    """Median over visible training views of each point's class probabilities, and the visible-view count."""
    transforms = json.loads((scene / 'transforms_train.json').read_text())
    per_view, views_used = [], []
    for frame in transforms['frames']:
        name = Path(frame['file_path']).stem
        per_view.append(view_probabilities(scene, frame, transforms, probs_dir, points, tolerance))
        views_used.append({'name': name, 'visible_cells': int(np.isfinite(per_view[-1][:, 0]).sum()),
                           'probabilities_sha256': digest(probs_dir / f'{name}_probs.npy')})
    stack = np.stack(per_view)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # cells no view sees stay NaN
        return np.nanmedian(stack, axis=0), np.isfinite(stack[..., 0]).sum(axis=0), views_used


def run(config_path, export_dir, depth_points, truth_dir, scene, probs_dir, output, registration=None):
    import rasterio

    config = json.loads(Path(config_path).read_text())
    output, scene, probs_dir = Path(output), Path(scene), Path(probs_dir)
    if output.exists():
        raise ValueError('Use a new output directory')
    prediction = check_prediction(probs_dir, config)
    manifest = json.loads((Path(export_dir) / 'export-manifest.json').read_text())
    geo, shift = aligned_georeference(manifest['georeference'], registration)
    manifest = manifest | {'georeference': geo}
    truth = read_truth(truth_dir, 'JAX_068')
    shape = truth['dsm'].shape
    top, valid, filled, ground, hag = top_surface(depth_points, geo, truth, shift)
    points = cell_model_points(truth, filled, geo)
    tolerance = 1.0 / np.asarray(geo['model_to_enu'])[0][0]  # 1.0 m in model units

    probabilities, views, views_used = fuse_views(scene, probs_dir, points, tolerance)
    probabilities, views = probabilities.reshape(shape + (len(NAMES),)), views.reshape(shape)

    classes, model_only, label, known = classify(probabilities, views, hag, valid, config['classes']['raised_hag_m'],
                                                 config['scoring']['min_views'])
    splat_class, inside, surface, wall = splat_classes(manifest, truth, top, ground, classes, classes == CODES['building'])

    scored = (truth['classes'] != 65) & (truth['classes'] != 17)
    mapping = config['evaluation']['mapping']
    evaluation = evaluate(classes, truth['classes'], mapping, scored)
    diagnostic = evaluate(model_only, truth['classes'], mapping, scored)
    targets = config['evaluation']['targets']
    checks = {'overall_accuracy': evaluation['overall_accuracy'] >= targets['overall_accuracy_min'],
              'building_iou': (evaluation['building']['iou'] or 0) >= targets['building_iou_min'],
              'vegetation_iou': (evaluation['vegetation']['iou'] or 0) >= targets['vegetation_iou_min']}

    output.mkdir(parents=True)
    profile = {'driver': 'GTiff', 'width': shape[1], 'height': shape[0], 'count': 1, 'crs': 'EPSG:32617',
               'transform': rasterio.transform.from_origin(truth['west'], truth['north'], truth['resolution'], truth['resolution'])}
    for filename, array, dtype in (('class-raster.tif', classes, 'uint8'), ('model-only-raster.tif', model_only, 'uint8'),
                                   ('views.tif', views, 'uint8'), ('hag.tif', hag, 'float32')):
        with rasterio.open(output / filename, 'w', dtype=dtype, **profile) as sink:
            sink.write(array.astype(dtype), 1)
    np.savez_compressed(output / 'cell-probabilities.npz', probabilities=probabilities.astype(np.float16), names=NAMES)
    np.savez_compressed(output / 'splat-classes.npz', classes=splat_class, legend=json.dumps(CODES))
    preview(output / 'preview.png', classes, truth['classes'], scored, mapping)
    preview(output / 'preview-model-only.png', model_only, truth['classes'], scored, mapping)
    counts = lambda array: {name: int((array == code).sum()) for name, code in CODES.items()}
    report = {
        'schema_version': 1, 'config': str(config_path), 'config_sha256': digest(Path(config_path)),
        'export_manifest_sha256': digest(Path(export_dir) / 'export-manifest.json'),
        'depth_points_sha256': digest(Path(depth_points)), 'weights_sha256': prediction['weights_sha256'],
        'views': views_used, 'registration': str(registration) if registration else None, 'enu_shift_m': list(shift),
        'cells': {'total': int(classes.size), 'with_top_surface': int(valid.sum()), 'with_evidence': int(known.sum()),
                  'by_class': counts(classes), 'model_label': {n: int((known & (label == n)).sum()) for n in NAMES}},
        'splats': {'total': int(len(splat_class)), 'inside_grid': int(inside.sum()), 'surface': int(surface.sum()),
                   'walls': int(wall.sum()), 'by_class': counts(splat_class)},
        'evaluation': evaluation, 'model_only_diagnostic': diagnostic,
        'targets': targets, 'checks': checks, 'passed': all(checks.values()),
        'notes': ['No registration shift: labels are evaluated where the model places them (about 1.4 m horizontal offset).',
                  'The model-only variant ignores height and is a diagnostic; only the headline rules count.'],
    }
    (output / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default='configs/semantics-v5.json')
    parser.add_argument('--export', required=True)
    parser.add_argument('--points', required=True)
    parser.add_argument('--truth', required=True)
    parser.add_argument('--scene', required=True, help='Prepared Skyfall scene (masks/, transforms_train.json)')
    parser.add_argument('--probabilities', required=True, help='pipeline.landcover_predict output directory')
    parser.add_argument('--registration', help='dsm-validation.json whose measured horizontal offset is removed first')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = run(args.config, args.export, args.points, args.truth, args.scene, args.probabilities, args.output,
                 args.registration)
    e, d = report['evaluation'], report['model_only_diagnostic']
    print(json.dumps({'checks': report['checks'], 'passed': report['passed']}, indent=2))
    print(f"v5: overall accuracy {e['overall_accuracy']:.3f}; IoU building {e['building']['iou']}, vegetation {e['vegetation']['iou']}")
    print(f"Model only: overall accuracy {d['overall_accuracy']:.3f}; IoU building {d['building']['iou']}, vegetation {d['vegetation']['iou']}")


if __name__ == '__main__':
    main()

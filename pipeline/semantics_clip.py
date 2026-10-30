"""Semantics v3: LangSplat SAM + CLIP image features fused onto the DFC2019 grid (configs/semantics-v3.json).

Run with .venv-imagery after LangSplat's preprocess.py has written <name>_f.npy / <name>_s.npy for each training view
and scripts/clip-prototypes.py has encoded the config's prompts (see docs/SEMANTICS.md). No OSM, no NDVI.
"""

import argparse
import json
from pathlib import Path
import warnings

import numpy as np

from pipeline.dsm_validation import read_truth
from pipeline.phase2 import digest
from pipeline.semantics import CODES, cell_lonlat, evaluate, preview, splat_classes, top_surface

LEVELS = ('default', 's', 'm', 'l')


def load_prototypes(path, config):
    data = np.load(path)
    names = [str(n) for n in data['names']]
    if names != list(config['prompts']) or json.loads(str(data['prompts'])) != config['prompts']:
        raise ValueError('Prototypes were encoded from different prompts than the config')
    return names, data['vectors'].astype(np.float32)


def segment_probabilities(features, prototypes, scale=100.0):
    """Softmax over classes of scaled cosine similarity, one row per SAM segment."""
    features = features / np.linalg.norm(features, axis=1, keepdims=True).clip(1e-8)
    logits = scale * features @ prototypes.T
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def sample_probabilities(seg_maps, probabilities, rows, cols, level_weights):
    """Level-weighted class probabilities at mask pixels; NaN where no level has a segment."""
    weights = np.asarray([level_weights[level] for level in LEVELS], np.float32)
    ids = seg_maps[:, rows, cols]  # levels x samples
    known = ids >= 0
    sampled = probabilities[np.maximum(ids, 0)] * (known * weights[:, None])[..., None]
    total = (known * weights[:, None]).sum(axis=0)
    result = sampled.sum(axis=0) / total.clip(1e-8)[:, None]
    result[total == 0] = np.nan
    return result


def visible_pixels(u, v, depth, shape, coverage, tolerance):
    """Pixel indices of projected points that are in frame, covered and not hidden behind nearer points (z-buffer)."""
    from scipy.ndimage import minimum_filter

    height, width = shape
    cols, rows = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    inside = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height) & np.isfinite(depth)
    flat = rows[inside] * width + cols[inside]
    nearest = np.full(height * width, np.inf)
    np.minimum.at(nearest, flat, depth[inside])
    nearest = minimum_filter(nearest.reshape(shape), size=3, mode='constant', cval=np.inf).ravel()
    ok = np.zeros(len(u), bool)
    ok[inside] = depth[inside] <= nearest[flat] + tolerance
    if coverage is not None:
        ok[ok] = coverage[rows[ok], cols[ok]] > 0
    return ok, rows, cols


def classify_clip(probabilities, views, hag, valid, names, raised_m, min_views):
    """v3 rules (height separates raised from flat) and the CLIP-only diagnostic."""
    known = valid & (views >= min_views) & np.isfinite(probabilities).all(axis=-1)
    best = np.argmax(np.nan_to_num(probabilities, nan=-1.0), axis=-1)
    label = np.asarray(names)[best]
    raised = hag >= raised_m
    rules = {
        'tree': (CODES['vegetation'], CODES['vegetation']),
        'grass': (CODES['grass'], CODES['grass']),
        'building': (CODES['building'], CODES['ground']),
        'road': (CODES['unknown'], CODES['road']), 'car': (CODES['unknown'], CODES['road']),
        'pavement': (CODES['unknown'], CODES['ground']), 'soil': (CODES['unknown'], CODES['ground']),
        'shadow': (CODES['unknown'], CODES['ground']),
        'water': (CODES['unknown'], CODES['water']),
    }
    clip_only_codes = {'tree': 'vegetation', 'grass': 'grass', 'building': 'building', 'road': 'road', 'car': 'road',
                       'pavement': 'ground', 'soil': 'ground', 'shadow': 'ground', 'water': 'water'}
    classes = np.zeros(hag.shape, np.uint8)
    clip_only = np.zeros(hag.shape, np.uint8)
    for name, (high, low) in rules.items():
        here = known & (label == name)
        classes[here] = np.where(raised[here], high, low)
        clip_only[here] = CODES[clip_only_codes[name]]
    return classes, clip_only, label, known


def cell_model_points(truth, filled, geo):
    """Model-frame coordinates of each cell's top-surface point (gap-filled height)."""
    from pyproj import Transformer

    lon, lat = cell_lonlat(truth)
    ecef = np.stack(Transformer.from_crs(4979, 4978, always_xy=True).transform(lon, lat, filled.ravel()), 1)
    frame = np.asarray(geo['enu_to_ecef'])
    enu = (ecef - frame[:3, 3]) @ frame[:3, :3]
    model_to_enu = np.asarray(geo['model_to_enu'])
    return (enu - model_to_enu[:3, 3]) @ np.linalg.inv(model_to_enu[:3, :3]).T


def view_probabilities(scene, name, features_dir, points, segment_probs_for, level_weights, tolerance):
    from PIL import Image

    camera = json.loads((scene / 'cameras' / f'{name}.json').read_text())
    P = (np.asarray(camera['K']).reshape(4, 4) @ np.asarray(camera['W2C']).reshape(4, 4))[:3]
    projected = np.column_stack((points, np.ones(len(points)))) @ P.T
    depth = projected[:, 2]
    width, height = camera['img_size']
    seg_maps = np.load(features_dir / f'{name}_s.npy').astype(np.int64)
    probabilities = segment_probs_for(np.load(features_dir / f'{name}_f.npy'))
    mask_height, mask_width = seg_maps.shape[1:]
    u = projected[:, 0] / depth * mask_width / width
    v = projected[:, 1] / depth * mask_height / height
    coverage = np.asarray(Image.fromarray(np.load(scene / 'masks' / f'{name}.npy')).resize((mask_width, mask_height), Image.NEAREST))
    ok, rows, cols = visible_pixels(u, v, np.where(depth > 0, depth, np.nan), (mask_height, mask_width), coverage, tolerance)
    result = np.full((len(points), probabilities.shape[1]), np.nan, np.float32)
    result[ok] = sample_probabilities(seg_maps, probabilities, rows[ok], cols[ok], level_weights)
    return result


def run(config_path, export_dir, depth_points, truth_dir, scene, features_dir, prototypes_path, output):
    import rasterio

    config = json.loads(Path(config_path).read_text())
    output, scene, features_dir = Path(output), Path(scene), Path(features_dir)
    if output.exists():
        raise ValueError('Use a new output directory')
    manifest = json.loads((Path(export_dir) / 'export-manifest.json').read_text())
    geo = manifest['georeference']
    truth = read_truth(truth_dir, 'JAX_068')
    shape = truth['dsm'].shape
    top, valid, filled, ground, hag = top_surface(depth_points, geo, truth)
    names, prototypes = load_prototypes(prototypes_path, config)
    points = cell_model_points(truth, filled, geo)
    scoring = config['scoring']
    tolerance = 1.0 / np.asarray(geo['model_to_enu'])[0][0]  # 1.0 m in model units

    frames = json.loads((scene / 'transforms_train.json').read_text())['frames']
    per_view, views_used = [], []
    for frame in frames:
        name = Path(frame['file_path']).stem
        per_view.append(view_probabilities(scene, name, features_dir, points, lambda f: segment_probabilities(f, prototypes),
                                           config['features']['level_weights'], tolerance))
        views_used.append({'name': name, 'visible_cells': int(np.isfinite(per_view[-1][:, 0]).sum()),
                           'features_sha256': digest(features_dir / f'{name}_f.npy')})
    stack = np.stack(per_view)
    views = np.isfinite(stack[..., 0]).sum(axis=0).reshape(shape)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # cells no view sees stay NaN
        probabilities = np.nanmedian(stack, axis=0).reshape(shape + (len(names),))

    classes, clip_only, label, known = classify_clip(probabilities, views, hag, valid, names,
                                                     config['classes']['raised_hag_m'], scoring['min_views'])
    splat_class, inside, surface, wall = splat_classes(manifest, truth, top, ground, classes, classes == CODES['building'])

    scored = (truth['classes'] != 65) & (truth['classes'] != 17)
    mapping = config['evaluation']['mapping']
    evaluation = evaluate(classes, truth['classes'], mapping, scored)
    diagnostic = evaluate(clip_only, truth['classes'], mapping, scored)
    targets = config['evaluation']['targets']
    checks = {'overall_accuracy': evaluation['overall_accuracy'] >= targets['overall_accuracy_min'],
              'building_iou': (evaluation['building']['iou'] or 0) >= targets['building_iou_min'],
              'vegetation_iou': (evaluation['vegetation']['iou'] or 0) >= targets['vegetation_iou_min']}

    output.mkdir(parents=True)
    profile = {'driver': 'GTiff', 'width': shape[1], 'height': shape[0], 'count': 1, 'crs': 'EPSG:32617',
               'transform': rasterio.transform.from_origin(truth['west'], truth['north'], truth['resolution'], truth['resolution'])}
    for filename, array, dtype in (('class-raster.tif', classes, 'uint8'), ('clip-only-raster.tif', clip_only, 'uint8'),
                                   ('views.tif', views, 'uint8'), ('hag.tif', hag, 'float32')):
        with rasterio.open(output / filename, 'w', dtype=dtype, **profile) as sink:
            sink.write(array.astype(dtype), 1)
    np.savez_compressed(output / 'cell-probabilities.npz', probabilities=probabilities.astype(np.float16), names=names)
    np.savez_compressed(output / 'splat-classes.npz', classes=splat_class, legend=json.dumps(CODES))
    preview(output / 'preview.png', classes, truth['classes'], scored, mapping)
    preview(output / 'preview-clip-only.png', clip_only, truth['classes'], scored, mapping)
    counts = lambda array: {name: int((array == code).sum()) for name, code in CODES.items()}
    report = {
        'schema_version': 1, 'config': str(config_path), 'config_sha256': digest(Path(config_path)),
        'export_manifest_sha256': digest(Path(export_dir) / 'export-manifest.json'),
        'depth_points_sha256': digest(Path(depth_points)), 'prototypes_sha256': digest(Path(prototypes_path)),
        'views': views_used,
        'cells': {'total': int(classes.size), 'with_top_surface': int(valid.sum()), 'with_evidence': int(known.sum()),
                  'by_class': counts(classes), 'clip_label': {n: int((known & (label == n)).sum()) for n in names}},
        'splats': {'total': int(len(splat_class)), 'inside_grid': int(inside.sum()), 'surface': int(surface.sum()),
                   'walls': int(wall.sum()), 'by_class': counts(splat_class)},
        'evaluation': evaluation, 'clip_only_diagnostic': diagnostic,
        'targets': targets, 'checks': checks, 'passed': all(checks.values()),
        'notes': ['No registration shift: labels are evaluated where the model places them (about 1.4 m horizontal offset).',
                  'The CLIP-only variant ignores height and is a diagnostic; only the headline rules count.'],
    }
    (output / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default='configs/semantics-v3.json')
    parser.add_argument('--export', required=True)
    parser.add_argument('--points', required=True)
    parser.add_argument('--truth', required=True)
    parser.add_argument('--scene', required=True, help='Prepared Skyfall scene (cameras/, masks/, transforms_train.json)')
    parser.add_argument('--features', required=True, help='LangSplat language_features directory')
    parser.add_argument('--prototypes', required=True, help='Output of scripts/clip-prototypes.py')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = run(args.config, args.export, args.points, args.truth, args.scene, args.features, args.prototypes, args.output)
    e, d = report['evaluation'], report['clip_only_diagnostic']
    print(json.dumps({'checks': report['checks'], 'passed': report['passed']}, indent=2))
    print(f"v3: overall accuracy {e['overall_accuracy']:.3f}; IoU building {e['building']['iou']}, vegetation {e['vegetation']['iou']}")
    print(f"CLIP only: overall accuracy {d['overall_accuracy']:.3f}; IoU building {d['building']['iou']}, vegetation {d['vegetation']['iou']}")


if __name__ == '__main__':
    main()

"""Per-splat semantic classes for a pipeline release, measured against DFC2019 labels (configs/semantics-v1.json).

Run with .venv-imagery. Works on the DFC2019 truth grid: the release's rendered-depth top surface, height above a
robust ground, calibrated WorldView-3 NDVI from the most nadir training views, and OpenStreetMap footprints.
"""

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import tarfile

import numpy as np

from pipeline.dsm_validation import enu_to_utm, points_dsm, read_ply, read_truth
from pipeline.phase2 import digest

ROOT = Path(__file__).resolve().parents[1]
CODES = {'unknown': 0, 'ground': 1, 'road': 2, 'building': 3, 'vegetation': 4, 'water': 5, 'grass': 6}
COLOURS = {0: (40, 40, 40), 1: (190, 180, 150), 2: (110, 110, 120), 3: (220, 80, 60), 4: (60, 170, 70), 5: (60, 120, 220),
           6: (150, 200, 110)}
DFC_COLOURS = {2: COLOURS[1], 5: COLOURS[4], 6: COLOURS[3], 9: COLOURS[5], 17: (200, 200, 60), 65: (0, 0, 0)}
MSI_BANDS = {'red': ('BAND_R', 5), 'nir1': ('BAND_N', 7)}


def parse_imd(text):
    """Sun elevation, first line time and per-band calibration from a Maxar .IMD."""
    bands = {}
    for name, body in re.findall(r'BEGIN_GROUP = (BAND_\w+)(.*?)END_GROUP = \1', text, re.S):
        values = dict(re.findall(r'(\w+) = ([^;]+);', body))
        bands[name] = {'absCalFactor': float(values['absCalFactor']), 'effectiveBandwidth': float(values['effectiveBandwidth'])}
    first = re.search(r'firstLineTime = ([^;]+);', text).group(1).strip()
    return {'sun_elevation_deg': float(re.search(r'meanSunEl = ([^;]+);', text).group(1)),
            'first_line_time': first, 'bands': bands}


def earth_sun_distance(when):
    """Astronomical unit distance for a UTC datetime (Meeus low-precision formula)."""
    julian = when.timestamp() / 86400 + 2440587.5
    g = math.radians(357.529 + 0.98560028 * (julian - 2451545.0))
    return 1.00014 - 0.01671 * math.cos(g) - 0.00014 * math.cos(2 * g)


def toa_reflectance(dn, band, imd, esun):
    radiance = dn.astype(np.float64) * band['absCalFactor'] / band['effectiveBandwidth']
    when = datetime.fromisoformat(imd['first_line_time'].replace('Z', '+00:00'))
    zenith = math.radians(90 - imd['sun_elevation_deg'])
    return math.pi * radiance * earth_sun_distance(when) ** 2 / (esun * math.cos(zenith))


def msi_imd(catalog_id, sources):
    product = next(p for p in sources['products'] if p['catalog_id'] == catalog_id and p['sensor_product'] == 'MSI')
    uri = product['uri'].rsplit('.', 1)[0] + '.tar'
    archive = ROOT / '.cache/core3d-metadata' / (hashlib.sha256(uri.encode()).hexdigest() + '.tar')
    with tarfile.open(archive) as package:
        member = next(m for m in package.getmembers() if m.name.endswith('.IMD') and '_MUL/' in m.name)
        text = package.extractfile(member).read().decode()
    return parse_imd(text), {'archive': str(archive.relative_to(ROOT)), 'sha256': digest(archive), 'member': member.name}


def view_ndvi(crop, imd, esun, lon, lat, height):
    import rasterio
    from rasterio.transform import RPCTransformer

    with rasterio.open(crop) as source:
        with RPCTransformer(source.rpcs) as transform:
            rows, cols = transform.rowcol(lon, lat, zs=height)
        rows, cols = np.asarray(rows), np.asarray(cols)
        inside = (rows >= 0) & (rows < source.height) & (cols >= 0) & (cols < source.width)
        red = source.read(MSI_BANDS['red'][1]); nir = source.read(MSI_BANDS['nir1'][1])
    result = np.full(len(lon), np.nan)
    r = toa_reflectance(red[rows[inside], cols[inside]], imd['bands'][MSI_BANDS['red'][0]], imd, esun['red'])
    n = toa_reflectance(nir[rows[inside], cols[inside]], imd['bands'][MSI_BANDS['nir1'][0]], imd, esun['nir1'])
    with np.errstate(invalid='ignore', divide='ignore'):
        result[inside] = (n - r) / (n + r)
    return result


def osm_masks(osm, grid, half_widths, shape):
    """Building, water and road-buffer masks on the truth grid from Overpass JSON with geometry."""
    from affine import Affine
    from pyproj import Transformer
    from rasterio.features import rasterize
    from scipy.ndimage import distance_transform_edt

    to_utm = Transformer.from_crs(4326, 32617, always_xy=True)
    transform = Affine(grid['resolution'], 0, grid['west'], 0, -grid['resolution'], grid['north'])

    def ring(points):
        x, y = to_utm.transform([p['lon'] for p in points], [p['lat'] for p in points])
        return list(zip(x, y))

    def polygons(selector):
        outers, inners = [], []
        for element in osm['elements']:
            tags = element.get('tags', {})
            if not selector(tags):
                continue
            if element['type'] == 'way' and element.get('geometry') and len(element['geometry']) >= 4:
                outers.append({'type': 'Polygon', 'coordinates': [ring(element['geometry'])]})
            elif element['type'] == 'relation':
                for member in element.get('members', []):
                    if member.get('geometry') and len(member['geometry']) >= 4:
                        shape_ = {'type': 'Polygon', 'coordinates': [ring(member['geometry'])]}
                        (inners if member.get('role') == 'inner' else outers).append(shape_)
        mask = rasterize(outers, out_shape=shape, transform=transform, fill=0, default_value=1).astype(bool) if outers else np.zeros(shape, bool)
        if inners:
            mask &= ~rasterize(inners, out_shape=shape, transform=transform, fill=0, default_value=1).astype(bool)
        return mask

    building = polygons(lambda t: 'building' in t)
    water = polygons(lambda t: t.get('natural') == 'water')
    road = np.zeros(shape, bool)
    by_width = {}
    for element in osm['elements']:
        kind = element.get('tags', {}).get('highway')
        if kind and element['type'] == 'way' and element.get('geometry'):
            width = half_widths.get(kind, half_widths['other'])
            by_width.setdefault(width, []).append({'type': 'LineString', 'coordinates': ring(element['geometry'])})
    for width, lines in by_width.items():
        centre = rasterize(lines, out_shape=shape, transform=transform, fill=0, default_value=1, all_touched=True)
        road |= distance_transform_edt(centre == 0) * grid['resolution'] <= width
    return building, water, road


# v1 thresholds; a config's "thresholds" entry overrides them. tree_hag_m splits vegetation into trees and grass.
DEFAULT_THRESHOLDS = {'low_hag_m': 1.0, 'building_hag_m': 2.5, 'vegetation_ndvi': 0.30, 'building_max_ndvi': 0.20,
                      'tree_hag_m': None}


def classify(hag, ndvi, valid, building, water, road, thresholds=None):
    t = DEFAULT_THRESHOLDS | (thresholds or {})
    classes = np.zeros(hag.shape, np.uint8)
    unmapped = np.zeros(hag.shape, bool)
    low = hag < t['low_hag_m']
    green = valid & (ndvi >= t['vegetation_ndvi'])
    rules = [('water', valid & water & low)]
    if t['tree_hag_m'] is None:
        rules.append(('vegetation', green))
    else:
        rules += [('vegetation', green & (hag >= t['tree_hag_m'])), ('grass', green)]
    rules += [
        ('building', valid & (hag >= t['building_hag_m']) & (building | ((ndvi < t['building_max_ndvi']) & ~road))),
        ('road', valid & road & low),
        ('ground', valid & low),
    ]
    assigned = np.zeros(hag.shape, bool)
    for name, mask in rules:
        mask = mask & ~assigned
        classes[mask] = CODES[name]
        if name == 'building':
            unmapped = mask & ~building
        assigned |= mask
    return classes, unmapped


def evaluate(classes, truth_classes, mapping, valid_truth):
    """Strict scores: our unknown cells count as incorrect."""
    ours = np.full(classes.shape, -1)
    for name, code in CODES.items():
        if mapping.get(name) is not None:
            ours[classes == code] = mapping[name]
    scored = valid_truth
    truth = truth_classes
    result = {'scored_cells': int(scored.sum()), 'unknown_cells': int((scored & (ours < 0)).sum()),
              'overall_accuracy': float((ours[scored] == truth[scored]).mean())}
    result['unknown_fraction'] = result['unknown_cells'] / result['scored_cells']
    for label, code in (('ground', 2), ('vegetation', 5), ('building', 6), ('water', 9)):
        t, o = scored & (truth == code), scored & (ours == code)
        union = (t | o).sum()
        result[label] = {'truth_cells': int(t.sum()), 'predicted_cells': int(o.sum()),
                         'iou': float((t & o).sum() / union) if union else None,
                         'precision': float((t & o).sum() / o.sum()) if o.sum() else None,
                         'recall': float((t & o).sum() / t.sum()) if t.sum() else None}
    labels = [2, 5, 6, 9, -1]
    result['confusion'] = {'rows_truth_cols_predicted': labels,
                           'counts': [[int((scored & (truth == a) & (ours == b)).sum()) for b in labels] for a in labels[:-1]]}
    return result


def preview(path, classes, truth_classes, scored, mapping):
    from PIL import Image

    def paint(values, colours):
        image = np.zeros(values.shape + (3,), np.uint8)
        for code, colour in colours.items():
            image[values == code] = colour
        return image

    ours = paint(classes, COLOURS)
    truth = paint(truth_classes, DFC_COLOURS)
    mapped = np.full(classes.shape, -1)
    for name, code in CODES.items():
        if mapping.get(name) is not None:
            mapped[classes == code] = mapping[name]
    disagree = np.zeros(classes.shape + (3,), np.uint8)
    disagree[scored] = (230, 230, 230)
    disagree[scored & (mapped != truth_classes)] = (220, 40, 40)
    Image.fromarray(np.concatenate([ours, truth, disagree], 1)).save(path)


def top_surface(depth_points, geo, truth, enu_shift=(0.0, 0.0, 0.0)):
    """Top surface, its gap-filled copy, robust ground and height above ground on the truth grid.

    enu_shift (metres) is added to the depth points, e.g. to remove a measured placement offset.
    """
    from scipy.ndimage import distance_transform_edt, percentile_filter

    top = points_dsm(enu_to_utm(np.load(depth_points)['enu'] + np.asarray(enu_shift), geo, 32617), truth)
    valid = np.isfinite(top)
    nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = top[tuple(nearest)]
    ground = percentile_filter(filled, 5, size=51)
    return top, valid, filled, ground, top - ground


def cell_lonlat(truth):
    from pyproj import Transformer

    rows, cols = np.indices(truth['dsm'].shape)
    east = truth['west'] + (cols.ravel() + 0.5) * truth['resolution']
    north = truth['north'] - (rows.ravel() + 0.5) * truth['resolution']
    return Transformer.from_crs(32617, 4326, always_xy=True).transform(east, north)


def splat_classes(manifest, truth, top, ground, classes, building):
    """Per-splat classes in fused-PLY order: surface splats take their cell class, walls inside buildings are building."""
    geo, shape = manifest['georeference'], truth['dsm'].shape
    vertices = read_ply(Path(manifest['fused_ply']['path']))
    model = np.stack([vertices[k] for k in 'xyz'], 1).astype(np.float64)
    transform = np.asarray(geo['model_to_enu'])
    splats = enu_to_utm(model @ transform[:3, :3].T + transform[:3, 3], geo, 32617)
    c = np.floor((splats[:, 0] - truth['west']) / truth['resolution']).astype(int)
    r = np.floor((truth['north'] - splats[:, 1]) / truth['resolution']).astype(int)
    inside = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
    splat_class = np.zeros(len(splats), np.uint8)
    ri, ci = r[inside], c[inside]
    surface = splats[inside, 2] >= top[ri, ci] - 1.5
    wall = ~surface & building[ri, ci] & (splats[inside, 2] - ground[ri, ci] >= 1.0)
    values = np.where(surface, classes[ri, ci], np.where(wall, CODES['building'], CODES['unknown']))
    splat_class[inside] = np.where(np.isfinite(top[ri, ci]) | wall, values, CODES['unknown'])
    return splat_class, inside, surface, wall


def run(config_path, export_dir, depth_points, truth_dir, osm_path, imagery_dir, sources_path, output):
    import rasterio

    config = json.loads(Path(config_path).read_text())
    output = Path(output)
    if output.exists():
        raise ValueError('Use a new output directory')
    manifest = json.loads((Path(export_dir) / 'export-manifest.json').read_text())
    geo = manifest['georeference']
    truth = read_truth(truth_dir, 'JAX_068')
    shape = truth['dsm'].shape
    top, valid, filled, ground, hag = top_surface(depth_points, geo, truth)
    lon, lat = cell_lonlat(truth)
    heights = np.nan_to_num(filled.ravel())
    sources = json.loads(Path(sources_path).read_text())
    esun = config['ndvi']['esun_thuillier_w_m2_um']
    per_view, calibration = [], {}
    for view in config['ndvi']['views']:
        imd, archive = msi_imd(view, sources)
        calibration[view] = {**archive, 'sun_elevation_deg': imd['sun_elevation_deg'], 'first_line_time': imd['first_line_time'],
                             'red': imd['bands']['BAND_R'], 'nir1': imd['bands']['BAND_N']}
        per_view.append(view_ndvi(Path(imagery_dir) / 'crops' / f'{view}-msi.tif', imd, esun, lon, lat, heights))
    with np.errstate(all='ignore'):
        ndvi = np.nanmedian(np.stack(per_view), axis=0).reshape(shape)

    osm = json.loads(Path(osm_path).read_text())
    building, water, road = osm_masks(osm, truth, config['classes']['road_half_widths_m'], shape)
    classes, unmapped = classify(hag, np.nan_to_num(ndvi, nan=0.0), valid, building, water, road,
                                 config['classes'].get('thresholds'))

    splat_class, inside, surface, wall = splat_classes(manifest, truth, top, ground, classes, building)

    scored = (truth['classes'] != 65) & (truth['classes'] != 17)
    mapping = config['evaluation']['mapping']
    evaluation = evaluate(classes, truth['classes'], mapping, scored)
    targets = config['evaluation']['targets']
    checks = {'overall_accuracy': evaluation['overall_accuracy'] >= targets['overall_accuracy_min'],
              'building_iou': (evaluation['building']['iou'] or 0) >= targets['building_iou_min'],
              'vegetation_iou': (evaluation['vegetation']['iou'] or 0) >= targets['vegetation_iou_min']}

    output.mkdir(parents=True)
    profile = {'driver': 'GTiff', 'width': shape[1], 'height': shape[0], 'count': 1, 'crs': 'EPSG:32617',
               'transform': rasterio.transform.from_origin(truth['west'], truth['north'], truth['resolution'], truth['resolution'])}
    for name, array, dtype in (('class-raster.tif', classes, 'uint8'), ('ndvi.tif', ndvi, 'float32'), ('hag.tif', hag, 'float32')):
        with rasterio.open(output / name, 'w', dtype=dtype, **profile) as sink:
            sink.write(array.astype(dtype), 1)
    np.savez_compressed(output / 'splat-classes.npz', classes=splat_class, legend=json.dumps(CODES))
    preview(output / 'preview.png', classes, truth['classes'], scored, mapping)
    counts = lambda array, mask=None: {name: int(((array == code) if mask is None else ((array == code) & mask)).sum()) for name, code in CODES.items()}
    report = {
        'schema_version': 1, 'config': str(config_path), 'config_sha256': digest(Path(config_path)),
        'export_manifest_sha256': digest(Path(export_dir) / 'export-manifest.json'),
        'depth_points_sha256': digest(Path(depth_points)), 'osm_sha256': digest(Path(osm_path)),
        'calibration': calibration,
        'cells': {'total': int(classes.size), 'with_top_surface': int(valid.sum()), 'by_class': counts(classes),
                  'building_unmapped': int(unmapped.sum())},
        'splats': {'total': int(len(splat_class)), 'inside_grid': int(inside.sum()),
                   'surface': int(surface.sum()), 'walls': int(wall.sum()), 'occluded_unknown': int((~surface & ~wall).sum()),
                   'by_class': counts(splat_class)},
        'evaluation': evaluation, 'targets': targets, 'checks': checks, 'passed': all(checks.values()),
        'notes': ['OSM is a 2026 snapshot; the imagery is 2014-2015; DFC2019 label date is unpublished.',
                  'No registration shift: labels are evaluated where the model places them (about 1.4 m horizontal offset).'],
    }
    (output / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/semantics-v1.json')
    parser.add_argument('--export', required=True)
    parser.add_argument('--points', required=True)
    parser.add_argument('--truth', required=True)
    parser.add_argument('--osm', required=True)
    parser.add_argument('--imagery', required=True, help='Block inputs directory containing crops/')
    parser.add_argument('--sources', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = run(args.config, args.export, args.points, args.truth, args.osm, args.imagery, args.sources, args.output)
    print(json.dumps({k: report[k] for k in ('cells', 'checks', 'passed')}, indent=2))
    e = report['evaluation']
    print(f"overall accuracy {e['overall_accuracy']:.3f}; IoU building {e['building']['iou']}, vegetation {e['vegetation']['iou']}")


if __name__ == '__main__':
    main()

"""Build and qualify the east-neighbour 3DEP key on CPU, without registration.

2007 has no documented vertical datum: its heights are used ONLY in the native
relative stability comparison, never as absolute truth. All choices below are
fixed before qualification; DFC2019 supplies the grid alignment and test only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from pyproj import Transformer, enums, network
from pyproj.transformer import TransformerGroup

from pipeline.dsm_validation import enu_to_utm, error_stats, read_truth, truth_enu

GRID_HASH = 'fa9a407ac7ee3f5a3694008e4bcd09ce9cc250452f0c3b11700a4960340abce2'
EPOCH = 2010.0  # GEOID18's NAD83(2011) reference epoch, not flight date.
CLASS_NAMES = {'ground': 2, 'vegetation_or_unclassified_non_ground': 5,
               'building': 6, 'water': 9, 'bridge': 17}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_points(path):
    """Accept cached EPT NPZ or LAS/LAZ; coordinates must be EPSG:3857, z metres."""
    path = Path(path)
    if path.suffix == '.npz':
        with np.load(path) as source:
            return {key: source[key] for key in ('x', 'y', 'z', 'classification')}
    import laspy
    las = laspy.read(path)
    return {key: np.asarray(getattr(las, key)) for key in ('x', 'y', 'z', 'classification')}


class HeightConversion:
    def __init__(self, grid):
        network.set_network_enabled(False)
        grid = Path(grid).resolve()
        if not grid.is_file():
            raise FileNotFoundError(f'Offline GEOID18 grid needed: {grid}')
        if digest(grid) != GRID_HASH:
            raise ValueError('Unexpected GEOID18 grid checksum')
        self.geoid = Transformer.from_pipeline(
            '+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad '
            f'+step +proj=vgridshift +grids={grid} +multiplier=1 '
            '+step +proj=unitconvert +xy_in=rad +xy_out=deg')
        group = TransformerGroup(6319, 7665, always_xy=True, allow_ballpark=False)
        if not group.best_available or not group.transformers:
            missing = [g.short_name for op in group.unavailable_operations for g in op.grids if not g.available]
            raise RuntimeError(f'Offline datum transformation unavailable; grids: {missing}')
        self.frame = group.transformers[0]
        if 'ITRF2008' not in self.frame.description:
            raise ValueError('Expected documented ITRF2008/WGS84(G1762) operation')
        self.to_lonlat = Transformer.from_crs(3857, 4326, always_xy=True)
        self.to_utm = Transformer.from_crs(4326, 32617, always_xy=True)
        self.provenance = {
            'geoid_grid': str(grid), 'sha256': digest(grid), 'network': 'OFF',
            'geoid_pipeline': self.geoid.definition,
            'nad83_to_wgs84_pipeline': self.frame.definition,
            'operation': self.frame.description, 'operation_projjson': self.frame.to_json_dict(),
            'coordinate_epoch': EPOCH, 'target_realisation': 'WGS84(G1762), EPSG:7665',
            'epoch_reason': 'GEOID18 gives NAD83(2011) epoch 2010.0 ellipsoid heights; no undocumented flight-epoch motion applied.',
            'horizontal': 'EPT already uses WGS84 EPSG:3857. Preserve its horizontal positions; inverse Helmert only locates NAD83 coordinates for geoid sampling and height conversion.',
            'horizontal_pipeline': self.to_utm.definition,
            'ept_to_wgs84_geographic_pipeline': self.to_lonlat.definition,
        }

    def heights(self, lon, lat, navd):
        """Recover NAD83 lat/lon at the same physical position, then h=H+N.

        Iterate the inverse 3D frame step because its latitude depends weakly on
        height. Horizontal output remains the original EPT WGS84 coordinates.
        """
        h = np.asarray(navd, dtype=float) - 30
        times = np.full(np.shape(h), EPOCH)
        for _ in range(3):
            nl, na, _, _ = self.frame.transform(lon, lat, h, times, direction=enums.TransformDirection.INVERSE)
            nh = self.geoid.transform(nl, na, navd, errcheck=True)[2]
            h = self.frame.transform(nl, na, nh, times, errcheck=True)[2]
        return np.asarray(h)

    def convert(self, points):
        lon, lat = self.to_lonlat.transform(points['x'], points['y'])
        x, y = self.to_utm.transform(lon, lat)
        return np.column_stack((x, y, self.heights(lon, lat, points['z'])))


def rasterize(xyz, classes, grid):
    """Highest non-noise point, all returns; class of that point, no filling."""
    size = grid['size']
    col = np.floor((xyz[:, 0] - grid['west']) / grid['resolution']).astype(int)
    row = np.floor((grid['north'] - xyz[:, 1]) / grid['resolution']).astype(int)
    valid = ((col >= 0) & (col < size) & (row >= 0) & (row < size)
             & np.isfinite(xyz).all(axis=1) & ~np.isin(classes, [7, 18]))
    indices = np.flatnonzero(valid)
    # Stable ordering gives a deterministic class when heights tie.
    indices = indices[np.argsort(xyz[indices, 2], kind='stable')]
    flat = row[indices] * size + col[indices]
    dsm = np.full(size * size, np.nan)
    cls = np.full(size * size, 65, dtype=np.uint8)
    count = np.bincount(flat, minlength=size * size).reshape(size, size)
    # Last occurrence of each sorted cell has maximum z.
    _, reverse_index = np.unique(flat[::-1], return_index=True)
    chosen = indices[len(indices) - 1 - reverse_index]
    locations = row[chosen] * size + col[chosen]
    dsm[locations], cls[locations] = xyz[chosen, 2], classes[chosen]
    return dsm.reshape(size, size), cls.reshape(size, size), count


def stable_mask(old, new, old_count, new_count, tolerance=0.5, minimum=1):
    return (np.isfinite(old) & np.isfinite(new) & (old_count >= minimum)
            & (new_count >= minimum) & (np.abs(new - old) <= tolerance))


def qualification(predicted, truth, labelled):
    stats = error_stats(predicted, truth, labelled)
    accepted = (stats['pixels'] > 0 and stats['mae_m'] <= 0.5 and abs(stats['median_m']) <= 0.3)
    return {'accepted': bool(accepted), 'limits': {'mae_m': 0.5, 'abs_median_m': 0.3},
            'all_labelled': stats}


def write_truth(directory, dsm, cls, grid):
    import rasterio
    from rasterio.transform import from_origin
    directory.mkdir()
    for suffix, array, nodata in [('DSM', dsm.astype('float32'), np.nan), ('CLS', cls.astype('uint8'), 65)]:
        with rasterio.open(directory / f'JAX_068_{suffix}.tif', 'w', driver='GTiff',
                           width=grid['size'], height=grid['size'], count=1,
                           dtype=array.dtype, crs='EPSG:32617', nodata=nodata,
                           transform=from_origin(grid['west'], grid['north'], grid['resolution'], grid['resolution']),
                           compress='deflate') as target:
            target.write(array, 1)
    np.savetxt(directory / 'JAX_068_DSM.txt', [grid['west'], grid['north'] - grid['size'] * grid['resolution'],
                                            grid['size'], grid['resolution']], fmt='%.12f')


def build(source, truth_dir, output, geoid_grid):
    for path in (Path('.'), Path(output).parent):
        if shutil.disk_usage(path).free < 20 * 1024**3:
            raise RuntimeError(f'Less than 20 GiB free: {path}')
    output = Path(output)
    if output.exists():
        raise ValueError('Use a fresh output directory')
    conversion = HeightConversion(geoid_grid)
    truth = read_truth(truth_dir, 'JAX_068')
    frame = json.loads(Path('configs/jax068-grid.json').read_text())
    corners = enu_to_utm(np.array([[e, n, h] for e in (-175, 425) for n in (-175, 175)
                                   for h in (-50, 300)]), frame, 32617)
    west = truth['west'] + np.floor((corners[:, 0].min() - truth['west']) / 0.5) * 0.5
    north = truth['north'] + np.ceil((corners[:, 1].max() - truth['north']) / 0.5) * 0.5
    size = int(np.ceil(max(corners[:, 0].max() - west, north - corners[:, 1].min()) / 2)) * 4
    grid = {'west': west, 'north': north, 'size': size, 'resolution': 0.5}
    coarse = {**grid, 'size': size // 4, 'resolution': 2.0}
    native, fine, ground, inputs = {}, {}, {}, {}
    for year, resource in [('2007', 'FL_DuvalCo_2007'), ('2018', 'FL_Peninsular_FDEM_Duval_2018')]:
        path = Path(source) / f'ept-{resource}' / 'points.npz'
        manifest = json.loads(path.with_name('manifest.json').read_text())
        if digest(path) != manifest['points_sha256']:
            raise ValueError(f'Input hash mismatch: {path}')
        points = read_points(path)
        lon, lat = conversion.to_lonlat.transform(points['x'], points['y'])
        x, y = conversion.to_utm.transform(lon, lat)
        xyz = np.column_stack((x, y, points['z']))
        classes = points['classification']
        native[year] = rasterize(xyz, classes, coarse)
        is_ground = classes == 2
        ground[year] = rasterize(xyz[is_ground], classes[is_ground], coarse)
        # 2007 is deliberately retained in its undocumented native height basis.
        if year == '2018':
            xyz[:, 2] = conversion.heights(lon, lat, points['z'])
        fine[year] = rasterize(xyz, classes, grid)
        inputs[year] = {'path': str(path), 'sha256': digest(path), 'points': len(xyz),
                        'manifest_sha256': digest(path.with_name('manifest.json'))}
    stable = stable_mask(native['2007'][0], native['2018'][0], native['2007'][2], native['2018'][2])
    stable_fine = stable.repeat(4, axis=0).repeat(4, axis=1)
    dsm, original_classes, _ = fine['2018']
    # Class 1 cannot be split reliably into trees/buildings. Use code 5 as an
    # explicit generic non-ground proxy rather than excluding tall class-1 roofs.
    classes = np.full_like(original_classes, 65)
    for code in (2, 5, 6, 9, 17):
        classes[original_classes == code] = code
    classes[original_classes == 1] = 5
    masked = classes.copy()
    masked[~stable_fine] = 65
    output.mkdir()
    write_truth(output / '2018', dsm, classes, grid)
    write_truth(output / 'stable', dsm, masked, grid)
    # Relative rasters are kept separate from dsm_validation-compatible truth.
    np.savez_compressed(output / 'relative-stability.npz', stable_2m=stable,
                        dsm_2007_native_2m=native['2007'][0], dsm_2018_navd88_2m=native['2018'][0],
                        count_2007_2m=native['2007'][2], count_2018_2m=native['2018'][2],
                        dsm_2007_native_05m=fine['2007'][0], classes_2007=fine['2007'][1])
    gs = stable & np.isfinite(ground['2007'][0]) & np.isfinite(ground['2018'][0])
    ground_difference = ground['2007'][0] - ground['2018'][0]
    ground_common = np.isfinite(ground_difference)
    coverage = {}
    key_truth = {**grid, 'dsm': dsm, 'classes': classes}
    enu = truth_enu(key_truth, frame['enu_to_ecef']).reshape(size, size, 3)
    e, n = enu[..., 0], enu[..., 1]
    regions = {'centre': (e >= -125) & (e < 125) & (n >= -125) & (n < 125),
               'east': (e >= 125) & (e < 375) & (n >= -125) & (n < 125),
               'seam_plus_minus_10m': (e >= 115) & (e < 135) & (n >= -125) & (n < 125)}
    regions['union'] = regions['centre'] | regions['east']
    for name, region in regions.items():
        coverage[name] = {}
        for label, code in [('all_labelled', None), *CLASS_NAMES.items()]:
            eligible = region & np.isfinite(dsm) & (classes != 65)
            if code is not None:
                eligible &= classes == code
            count = int((eligible & stable_fine).sum())
            denominator = int(eligible.sum())
            coverage[name][label] = {'stable_pixels': count, 'available_labelled_pixels': denominator,
                                    'fraction_of_available': count / denominator if denominator else None,
                                    'fraction_of_region': count / int(region.sum())}
    rr, cc = np.indices(truth['dsm'].shape)
    cols = np.rint((truth['west'] - west) / 0.5).astype(int) + cc
    rows = np.rint((north - truth['north']) / 0.5).astype(int) + rr
    comparison = {}
    # 2007 standalone diagnostics compare native heights to DFC transformed to
    # the 2018 NAVD88 basis. They do not establish a 2007 absolute datum.
    tx = truth['west'] + (cc + 0.5) * 0.5
    ty = truth['north'] - (rr + 0.5) * 0.5
    tlon, tlat = Transformer.from_crs(32617, 4326, always_xy=True).transform(tx, ty)
    correction = conversion.heights(tlon, tlat, np.zeros_like(tx))
    for name, predicted, labels, reference in [
        ('2018', dsm[rows, cols], classes[rows, cols], truth['dsm']),
        ('stable_2018', dsm[rows, cols], masked[rows, cols], truth['dsm']),
        ('2007_relative_only', fine['2007'][0][rows, cols], fine['2007'][1][rows, cols], truth['dsm'] - correction)]:
        labelled = (labels != 65) & (labels != 9) & (truth['classes'] != 65) & (truth['classes'] != 9)
        comparison[name] = {'all_labelled': error_stats(predicted, reference, labelled)}
        for label, code in {'ground': 2, 'vegetation': 5, 'building': 6, 'water': 9, 'bridge': 17}.items():
            comparison[name][label] = error_stats(predicted, reference, labelled & (truth['classes'] == code))
    result = qualification(dsm[rows, cols], truth['dsm'],
                           (masked[rows, cols] != 65) & (masked[rows, cols] != 9)
                           & (truth['classes'] != 65) & (truth['classes'] != 9))
    anchor = frame['anchor']
    lon, lat = anchor['longitude'], anchor['latitude']
    nad_h = conversion.geoid.transform(lon, lat, 0)[2]
    wgs_h = conversion.frame.transform(lon, lat, nad_h, EPOCH)[2]
    report = {'status': 'accepted' if result['accepted'] else 'rejected', 'qualification': result,
              'comparisons': comparison, 'coverage': coverage, 'grid': grid,
              'comparison_class_basis': 'Per-class comparisons use DFC2019 classes; coverage uses mapped 2018 classes.',
              'transformation_2018': conversion.provenance,
              'anchor_navd88_zero_nad83_ellipsoidal_m': nad_h,
              'anchor_navd88_zero_wgs84_ellipsoidal_m': wgs_h,
              'anchor_nad83_to_wgs84_height_change_m': wgs_h - nad_h,
              'datum_2007': 'Undocumented; ground consistent with 2018 NAVD88 to 0.12 m in input evidence. Relative stability only; no absolute conversion or offset removal.',
              'ground_2007_minus_2018_median_m': float(np.median(ground_difference[ground_common])),
              'ground_common_cells': int(ground_common.sum()),
              'stable_ground_2007_minus_2018_median_m': float(np.median(ground_difference[gs])),
              'stable_ground_cells': int(gs.sum()),
              'stability': {'cell_m': 2, 'minimum_points_each': 1, 'max_absolute_difference_m': 0.5,
                            'upsampling': 'Each qualifying 2 m cell supplies a constant mask to its 16 fine pixels; missing 0.5 m DSM pixels remain unlabelled.'},
              'class_mapping': {'2': '2 ground', '6': '6 building', '1': '5 generic unclassified non-ground proxy (includes buildings and vegetation; not a vegetation assertion)',
                                '5': '5 vegetation', '9': '9 water', '17': '17 bridge',
                                '7,18': 'noise excluded from DSM', 'other_or_missing': '65 excluded'},
              'rasterization': 'Highest non-noise point, all returns, no interpolation. Matches pipeline.dsm_validation.points_dsm max-cell convention. Original DFC2019 lidar-to-DSM production rule is not available locally; exact equivalence not established.',
              'inputs': inputs, 'dfc_files_sha256': {p.name: digest(p) for p in Path(truth_dir).iterdir() if p.is_file()},
              'metadata_sha256': {p.name: digest(p) for p in (Path(source) / 'metadata').glob('*') if p.is_file()},
              'sources': ['https://www.ngs.noaa.gov/GEOID/GEOID18/geoid18_tech_details.shtml',
                          'EPSG operation inverse 7807 (ITRF2008 to NAD83(2011)), inverse WGS84(G1762) to ITRF2008 (1); full PROJJSON recorded'],
              'next': 'Continue steps 4–8 only if accepted; rejected key must not grade models.',
              'training_executed': False}
    (output / 'qualification.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='artifacts/reference/usgs-3dep')
    parser.add_argument('--truth', default='artifacts/reference/dfc2019-jax068')
    parser.add_argument('--output', required=True)
    parser.add_argument('--geoid-grid', default=str(Path.home() / '.local/share/proj/us_noaa_g2018u0.tif'))
    args = parser.parse_args()
    result = build(args.source, args.truth, args.output, args.geoid_grid)
    print(json.dumps(result['qualification'], indent=2))
    return 0 if result['qualification']['accepted'] else 2


if __name__ == '__main__':
    raise SystemExit(main())

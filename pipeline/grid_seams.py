"""Seam and cell checks for adjacent cells of a multi-cell grid, against pre-registered rules.

Run after export, rendered depth and dsm_validation (with --grid and --cell) for every cell named in the rules file.
Measurements reuse pipeline.seam_check.seam_metrics, so seams are measured exactly as in the split experiment. A
horizontal seam is measured by rotating the shared-frame coordinates 90 degrees so that it becomes the vertical seam
seam_metrics expects; the rasters and validation reports are those of the real grid.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from pipeline.dsm_validation import read_ply, read_truth, truth_enu
from pipeline.ownership import cell_mask, grid_partition_check, in_frame
from pipeline.phase2 import digest
from pipeline.seam_check import seam_metrics


def seam_axis(grid, low, high):
    """0 if `low` is directly west of `high` (vertical seam), 1 if directly south (horizontal seam)."""
    a, b = grid['cells'][low], grid['cells'][high]
    if a['east_m'][1] == b['east_m'][0] and a['north_m'] == b['north_m']:
        return 0
    if a['north_m'][1] == b['north_m'][0] and a['east_m'] == b['east_m']:
        return 1
    raise ValueError(f'{low} must lie directly west or south of {high}')


def pair_frame(grid, low, high, points):
    """Points and a two-cell grid in which the seam between `low` (west) and `high` (east) is vertical."""
    axis = seam_axis(grid, low, high)
    swap = (lambda p: p[:, [1, 0, 2]]) if axis else (lambda p: p)
    cells = {name: grid['cells'][cell] for name, cell in (('west', low), ('east', high))}
    if axis:
        cells = {name: {'east_m': c['north_m'], 'north_m': c['east_m']} for name, c in cells.items()}
    return swap(np.asarray(points, dtype=float)), {'cells': cells}


def read_cell(grid, cell, export, depth, validation, truth_dir, shape):
    import rasterio

    manifest_path = Path(export) / 'export-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    report_path = Path(validation) / 'dsm-validation.json'
    report = json.loads(report_path.read_text())
    if (report['export_manifest_sha256'] != digest(manifest_path)
            or report['method'].get('depth_points_sha256') != digest(Path(depth))
            or report.get('ownership') != {'grid': grid, 'cell': cell}
            or report['reference']['files_sha256'] != {p.name: digest(p) for p in sorted(Path(truth_dir).iterdir()) if p.is_file()}):
        raise ValueError(f'{cell}: validation must match export, depth points, truth and ownership grid')
    rasters = {}
    for key, filename in (('raw', 'splat-dsm.tif'), ('registered', 'splat-dsm-registered.tif')):
        if digest(Path(validation) / filename) != report['outputs']['raster_sha256'][filename]:
            raise ValueError(f'{cell}: validation raster changed after measurement')
        with rasterio.open(Path(validation) / filename) as source:
            rasters[key] = source.read(1).astype(float)
        if rasters[key].shape != shape:
            raise ValueError(f'{cell}: validation raster differs from truth grid')
    vertices = read_ply(Path(manifest['fused_ply']['path']))
    geo = manifest['georeference']
    model = np.stack([vertices[k] for k in 'xyz'], 1).astype(float)
    transform = np.asarray(geo['model_to_enu'])
    centres = in_frame(model @ transform[:3, :3].T + transform[:3, 3], geo['enu_to_ecef'], grid['enu_to_ecef'])
    return {'report': report, 'rasters': rasters, 'centres': centres,
            'provenance': {'export_sha256': digest(manifest_path), 'depth_sha256': digest(Path(depth)),
                           'validation_sha256': digest(report_path)}}


def decide(cells, seams, baseline, rules):
    checks = {}
    for name, cell in cells.items():
        mae = cell['report']['registered']['all_labelled'].get('mae_m')
        coverage = cell['report']['coverage']['block_area_covered_fraction']
        checks[f'{name}_coverage'] = coverage >= rules['min_lidar_coverage_fraction']
        checks[f'{name}_mae'] = mae is not None and mae <= baseline[name] + rules['cell_mae_max_excess_over_baseline_m']
    for pair, seam in seams.items():
        checks[f'{pair}_ownership'] = seam['doubled_centres'] <= rules['ownership_doubled_centres_max']
        for side in ('west', 'east'):
            errors = seam['metrics']['lidar_error_by_distance'][side]
            checks[f'{pair}_{side}_band'] = (errors['seam']['pixels'] > 0 and errors['interior']['pixels'] > 0
                and errors['seam']['mae_m'] <= errors['interior']['mae_m'] + rules['seam_band_mae_max_excess_over_interior_m'])
        for kind, limit in (('ground', 'seam_residual_step_ground_max_abs_m'), ('all', 'seam_residual_step_all_max_abs_m')):
            step = seam['metrics']['height_continuity'][kind]['unregistered'].get('residual_step_m')
            checks[f'{pair}_step_{kind}'] = step is not None and abs(step) <= rules[limit]
    return {'passed': all(checks.values()), 'checks': checks}


def check(grid_path, decision_path, truth_dir, inputs):
    """inputs: {cell: {'export': dir, 'depth': npz, 'validation': dir}} for every cell in the rules' baseline."""
    grid = json.loads(Path(grid_path).read_text())
    decision = json.loads(Path(decision_path).read_text())
    if not grid_partition_check(grid)['passed']:
        raise ValueError('Grid ownership contains a gap or overlap')
    truth = read_truth(truth_dir, 'JAX_068')
    shared = truth_enu(truth, grid['enu_to_ecef'])
    baseline = decision['baseline']['registered_all_labelled_mae_m']
    cells = {name: read_cell(grid, name, *(inputs[name][k] for k in ('export', 'depth', 'validation')), truth_dir,
                             truth['dsm'].shape) for name in baseline}
    seams = {}
    for pair in decision['seams']:
        low, high = pair.split('|')
        # Seam pairs are named west|east or north|south; seam_metrics wants the west or south cell first.
        if seam_axis_safe(grid, high, low):
            low, high = high, low
        points, frame = pair_frame(grid, low, high, shared)
        metrics = seam_metrics({'west': cells[low]['rasters']['raw'], 'east': cells[high]['rasters']['raw']},
                               {'west': cells[low]['rasters']['registered'], 'east': cells[high]['rasters']['registered']},
                               truth, points, frame)
        axis = seam_axis(grid, low, high)
        boundary = grid['cells'][low]['north_m' if axis else 'east_m'][1]
        doubled = near = 0
        for name in (low, high):
            centres = cells[name]['centres']
            own_low, own_high = cell_mask(centres, grid['cells'][low]), cell_mask(centres, grid['cells'][high])
            doubled += int((own_low & own_high).sum())
            near += int(((own_low | own_high) & (np.abs(centres[:, axis] - boundary) <= 1)).sum())
        seams[pair] = {'west_or_south': low, 'east_or_north': high, 'axis': 'north' if axis else 'east',
                       'boundary_m': boundary, 'doubled_centres': doubled, 'centres_within_1m': near, 'metrics': metrics}
    return {'schema_version': 1, 'status': 'measured', 'grid_sha256': digest(Path(grid_path)),
            'decision_sha256': digest(Path(decision_path)),
            'cells': {name: {'provenance': c['provenance'], 'coverage': c['report']['coverage'],
                             'registered': c['report']['registered'], 'baseline_mae_m': baseline[name]}
                      for name, c in cells.items()},
            'seams': seams, 'decision': decide(cells, seams, baseline, decision['rules'])}


def seam_axis_safe(grid, low, high):
    try:
        seam_axis(grid, low, high)
        return True
    except ValueError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', required=True)
    parser.add_argument('--decision', required=True)
    parser.add_argument('--truth', required=True)
    parser.add_argument('--inputs', required=True, help='JSON: {cell: {export, depth, validation}}')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = check(args.grid, args.decision, args.truth, json.loads(Path(args.inputs).read_text()))
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['decision'], indent=2))
    raise SystemExit(0 if result['decision']['passed'] else 2)


if __name__ == '__main__':
    main()

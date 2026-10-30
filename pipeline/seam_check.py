"""Check split ownership, rendered-depth height continuity and registered lidar errors.

Run after export, rendered depth and dsm_validation for both blocks. The crop is
exactly the shared-grid crop used by pipeline.tiling. Gaussian kernel overlap is
not centre ownership; this report does not claim a visual mixed-LOD seam test.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from pipeline.dsm_validation import (read_truth, read_ply, truth_enu, error_stats)
from pipeline.ownership import in_frame, cell_mask, partition_check
from pipeline.phase2 import digest


def height_step(west, east, truth, west_strip, east_strip, mask):
    """Difference of strip medians (east minus west), with sample counts."""
    wm = west_strip & mask & np.isfinite(west) & np.isfinite(truth)
    em = east_strip & mask & np.isfinite(east) & np.isfinite(truth)
    result = {'west_pixels': int(wm.sum()), 'east_pixels': int(em.sum()), 'step_m': None}
    if wm.any() and em.any():
        result.update(step_m=float(np.median(east[em]) - np.median(west[wm])),
                      lidar_step_m=float(np.median(truth[em]) - np.median(truth[wm])),
                      residual_step_m=float(np.median(east[em]-truth[em]) - np.median(west[wm]-truth[wm])))
    return result


def seam_metrics(raw, registered, truth, shared_points, grid, band=10):
    shape = truth['dsm'].shape
    east, north = shared_points[:, :2].T
    seam = grid['cells']['west']['east_m'][1]
    masks = {name: cell_mask(shared_points, grid['cells'][name]).reshape(shape) for name in ('west','east')}
    distance = np.abs(east - seam).reshape(shape)
    labelled = (truth['classes'] != 65) & (truth['classes'] != 9)
    errors = {}
    for name in ('west','east'):
        errors[name] = {
            region: error_stats(registered[name], truth['dsm'], masks[name] & labelled & condition)
            for region, condition in [('seam', distance <= band), ('interior', distance > band)]}
    strips = {name: masks[name] & (distance <= band) for name in masks}
    heights = {}
    for kind, mask in [('ground', truth['classes'] == 2), ('all', np.ones(shape, bool))]:
        heights[kind] = {
            'unregistered': height_step(raw['west'], raw['east'], truth['dsm'], strips['west'], strips['east'], mask),
            'registered': height_step(registered['west'], registered['east'], truth['dsm'], strips['west'], strips['east'], mask)}
    return {'lidar_error_by_distance': errors, 'height_continuity': heights}


def decide(metrics, ownership, rules, block_mae):
    checks = {'ownership': ownership['partition']['passed'] and ownership['doubled_centres'] == 0
              and ownership['partition']['gap_width_m'] <= rules['ownership_gap_width_max_m']
              and ownership['partition']['overlap_width_m'] <= rules['ownership_overlap_width_max_m']}
    for name in ('west','east'):
        errors = metrics['lidar_error_by_distance'][name]
        checks[f'{name}_mae'] = block_mae[name] is not None and block_mae[name] <= rules['block_mae_max_m']
        checks[f'{name}_seam'] = (errors['seam']['pixels'] > 0 and errors['interior']['pixels'] > 0
            and errors['seam']['mae_m'] <= errors['interior']['mae_m'] + rules['seam_mae_max_excess_over_interior_m'])
    for kind in ('ground', 'all'):
        step = metrics['height_continuity'][kind]['unregistered']['step_m']
        checks[f'height_{kind}'] = step is not None and abs(step) <= rules[f'height_step_{kind}_max_abs_m']
    return {'passed': all(checks.values()), 'checks': checks}


def check(grid_path, exports, depths, validations, truth_dir, decision_path):
    import rasterio

    grid = json.loads(Path(grid_path).read_text())
    decision = json.loads(Path(decision_path).read_text())
    truth = read_truth(truth_dir, 'JAX_068')
    shared = truth_enu(truth, grid['enu_to_ecef'])
    partition = partition_check(grid)
    if not partition['passed']:
        raise ValueError('Grid ownership contains a gap or overlap')
    raw, registered, counts, provenance, block_mae = {}, {}, {}, {}, {}
    for name in ('west', 'east'):
        export = Path(exports[name])
        manifest_path = export / 'export-manifest.json'
        manifest = json.loads(manifest_path.read_text())
        fused = Path(manifest['fused_ply']['path'])
        if digest(fused) != manifest['fused_ply']['sha256']:
            raise ValueError('Changed fused PLY')
        vertices = read_ply(fused)
        model = np.stack([vertices[k] for k in 'xyz'], 1).astype(float)
        geo = manifest['georeference']
        transform = np.asarray(geo['model_to_enu'])
        enu = model @ transform[:3,:3].T + transform[:3,3]
        centres = in_frame(enu, geo['enu_to_ecef'], grid['enu_to_ecef'])
        own = cell_mask(centres, grid['cells'][name])
        other = cell_mask(centres, grid['cells']['east' if name == 'west' else 'west'])
        near = np.abs(centres[:,0] - partition['boundary_east_m']) <= 1
        counts[name] = {'input_centres': len(centres), 'owned_centres': int(own.sum()),
                        'owned_within_1m': int((own & near).sum()),
                        'doubled_after_crop': int((own & other).sum()),
                        'removed_within_1m': int((~own & near).sum())}
        validation = Path(validations[name])
        report_path = validation / 'dsm-validation.json'
        report = json.loads(report_path.read_text())
        if (report['export_manifest_sha256'] != digest(manifest_path)
            or report['method'].get('depth_points_sha256') != digest(Path(depths[name]))
            or report.get('ownership') != {'grid': grid, 'cell': name}
            or report['reference']['files_sha256'] != {p.name: digest(p) for p in sorted(Path(truth_dir).iterdir()) if p.is_file()}):
            raise ValueError('Validation must match export, depth points, truth and ownership grid')
        for target, filename in [(raw, 'splat-dsm.tif'), (registered, 'splat-dsm-registered.tif')]:
            if digest(validation / filename) != report['outputs']['raster_sha256'][filename]:
                raise ValueError('Validation raster changed after measurement')
            with rasterio.open(validation / filename) as source:
                target[name] = source.read(1).astype(float)
            if target[name].shape != truth['dsm'].shape:
                raise ValueError('Validation raster differs from truth grid')
        block_mae[name] = report['registered']['all_labelled'].get('mae_m')
        provenance[name] = {'export_sha256': digest(manifest_path), 'depth_sha256': digest(Path(depths[name])),
                            'validation_sha256': digest(report_path), 'coverage': report['coverage']}
    ownership = {'partition': partition, 'counts': counts,
                 'doubled_centres': sum(c['doubled_after_crop'] for c in counts.values()),
                 'scope': 'Shared half-open area partition and full-resolution centre crop; no claim of dense surface coverage or disjoint Gaussian kernels.'}
    metrics = seam_metrics(raw, registered, truth, shared, grid)
    return {'schema_version': 1, 'status': 'measured', 'grid_sha256': digest(Path(grid_path)),
            'decision_sha256': digest(Path(decision_path)), 'provenance': provenance,
            'ownership': ownership, **metrics, 'block_mae_m': block_mae,
            'decision': decide(metrics, ownership, decision['rules'], block_mae)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', default='configs/jax068-grid.json')
    parser.add_argument('--decision', default='configs/phase3-split-decision.json')
    parser.add_argument('--truth', default='artifacts/reference/dfc2019-jax068')
    for name in ('west','east'):
        for item in ('export','points','validation'):
            parser.add_argument(f'--{name}-{item}', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error('Use a new seam report path')
    result = check(args.grid, {n:getattr(args,n+'_export') for n in ('west','east')},
                   {n:getattr(args,n+'_points') for n in ('west','east')},
                   {n:getattr(args,n+'_validation') for n in ('west','east')}, args.truth, args.decision)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['decision'], indent=2))
    return 0 if result['decision']['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())

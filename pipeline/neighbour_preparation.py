"""Read-only central ownership audit and fail-closed neighbour Stage B startup."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from pipeline.ownership import cell_mask, in_frame, partition_check
from pipeline.phase2 import digest


def ownership_counts(points, georeference, grid):
    """Compare the delivered closed own-frame crop with the proposed shared crop."""
    old = (np.abs(points[:, 0]) <= 125) & (np.abs(points[:, 1]) <= 125)
    shared = in_frame(points, georeference['enu_to_ecef'], grid['enu_to_ecef'])
    centre = cell_mask(shared, grid['cells']['centre'])
    east = cell_mask(shared, grid['cells']['east'])
    delta = shared[:, 0] - 125
    edge = np.array([[125, north, up] for north in (-125, 125)
                     for up in (points[old, 2].min(), points[old, 2].max())])
    edge_east = in_frame(edge, georeference['enu_to_ecef'], grid['enu_to_ecef'])[:, 0]
    return {
        'old_closed_owned_centres': int(old.sum()),
        'old_owned_on_own_east_125_exact': int((old & (points[:, 0] == 125)).sum()),
        'old_owned_within_1m_own_east_125': int((old & (abs(points[:, 0] - 125) <= 1)).sum()),
        'old_owned_on_shared_east_125_exact': int((old & (delta == 0)).sum()),
        'old_owned_within_1m_shared_east_125': int((old & (abs(delta) <= 1)).sum()),
        'old_owned_in_neighbour_cell': int((old & east).sum()),
        'shared_east_max_old_owned_m': float(shared[old, 0].max()),
        'shared_north_range_old_owned_m': [float(shared[old, 1].min()), float(shared[old, 1].max())],
        'proposed_centre_owned_centres': int(centre.sum()),
        'proposed_removed_from_old': int((old & ~centre).sum()),
        'proposed_added_to_old': int((~old & centre).sum()),
        'proposed_doubled_centres': int((centre & east).sum()),
        'minimum_abs_east_seam_distance_old_owned_m': float(abs(delta[old]).min()),
        'old_east_edge_shared_east_range_m': [float(edge_east.min()), float(edge_east.max())],
        'old_east_edge_gap_to_125_range_m': [float(125 - edge_east.max()), float(125 - edge_east.min())],
        'partition': partition_check(grid),
    }


def audit_ownership(output):
    from pipeline.tiling import read_fused_ply, to_enu

    export = Path('artifacts/stage1/pilot-2015-19v-recipe-01/export-r2')
    grid_path = Path('configs/jax068-grid-neighbour.json')
    manifest = json.loads((export / 'export-manifest.json').read_text())
    plan = json.loads(Path('artifacts/tiles/pilot-2015-19v-recipe-01-r1/tile-plan.json').read_text())
    if (digest(export / 'export-manifest.json') != plan['export_manifest_sha256']
            or digest(Path(manifest['fused_ply']['path'])) != plan['fused_ply_sha256']):
        raise ValueError('Central release export no longer matches its tile plan')
    points = to_enu(read_fused_ply(manifest['fused_ply']['path']),
                    manifest['georeference']['model_to_enu'])['centres']
    counts = ownership_counts(points, manifest['georeference'], json.loads(grid_path.read_text()))
    if counts['old_closed_owned_centres'] != plan['crop']['owned_splats']:
        raise ValueError('Central crop count differs from the delivered release')
    result = dict(release_id='jax068-2015-19v-r1', export=str(export),
                  export_manifest_sha256=digest(export / 'export-manifest.json'),
                  fused_ply_sha256=digest(Path(manifest['fused_ply']['path'])),
                  grid_sha256=digest(grid_path), **counts,
                  scope='Full-resolution Gaussian centres; not LOD centroids or kernel support.')
    with Path(output).open('x') as target:
        target.write(json.dumps(result, indent=2) + '\n')
    return result


def require_stage_b(root=Path('.')):
    root = Path(root)
    for directory in (root, root / 'artifacts'):
        if shutil.disk_usage(directory).free < 20 * 1024**3:
            raise ValueError(f'Less than 20 GiB free: {directory}')
    qualification = root / 'artifacts/reference/3dep-jax068-east/qualification.json'
    if not qualification.is_file():
        raise ValueError(f'Missing qualified answer key: {qualification}')
    if json.loads(qualification.read_text()).get('accepted') is not True:
        raise ValueError(f'Answer key is not qualified (accepted must be true): {qualification}')
    decision = root / 'configs/neighbour-decision.json'
    if not decision.is_file():
        raise ValueError(f'Missing pre-registered decision: {decision}')
    if not isinstance(json.loads(decision.read_text()), dict):
        raise ValueError('Neighbour decision must be a JSON object')
    # Step 5 is deliberately deferred. Never substitute the split experiment's checker.
    checker = root / 'scripts/neighbour-seam-check.py'
    if not checker.is_file():
        raise ValueError(f'Deferred step 5 checker is not implemented: {checker}')
    for name in ('JAX_068_DSM.tif', 'JAX_068_DSM.txt', 'JAX_068_CLS.tif'):
        if not (qualification.parent / 'stable' / name).is_file():
            raise ValueError(f'Missing qualified truth payload: {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('ownership', 'stage-b-preflight'))
    parser.add_argument('--output', default='training/neighbour-ownership.json')
    args = parser.parse_args()
    try:
        if args.action == 'ownership':
            print(json.dumps(audit_ownership(args.output), indent=2))
        else:
            require_stage_b()
    except (ValueError, OSError) as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()

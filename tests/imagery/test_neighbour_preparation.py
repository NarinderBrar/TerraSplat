import json
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import pytest

from pipeline.neighbour_preparation import ownership_counts, require_stage_b
from pipeline.ownership import cell_mask, in_frame, partition_check

ROOT = Path(__file__).resolve().parents[2]


def read(name):
    return json.loads((ROOT / name).read_text())


def test_neighbour_preserves_acquisition_roles_and_entire_recipe():
    selection = read('configs/jax068-eastn-19v.json')
    original = read('configs/jax068-pilot-19v.json')
    assert selection.pop('block') == 'configs/jax068-eastn-block.json'
    original.pop('block')
    assert selection == original
    assert len(selection['train_catalog_ids']) == 19
    assert selection['holdout_catalog_ids'] == ['104001000A22A000']
    assert not set(selection['holdout_catalog_ids']) & set(selection['train_catalog_ids'])
    profile = read('configs/train-2015-eastn-19v-recipe.json')
    original = read('configs/train-2015-pilot-19v-recipe.json')
    assert profile.pop('id') != original.pop('id')
    assert profile == original


def test_grid_geometry_and_block_centre():
    grid = read('configs/jax068-grid-neighbour.json')
    assert grid['cells']['centre']['east_m'] == [-125, 125]
    assert grid['cells']['east']['east_m'] == [125, 375]
    assert all(c['north_m'] == [-125, 125] for c in grid['cells'].values())
    assert partition_check(grid)['passed']
    block = read('configs/jax068-eastn-block.json')
    pilot = read('configs/jax068-block.json')
    for key in ('side_m', 'halo_m', 'ellipsoidal_height_range_m'):
        assert block[key] == pilot[key]
    from pyproj import Transformer
    xyz = Transformer.from_crs(4979, 4978, always_xy=True).transform(
        block['longitude'], block['latitude'], 0)
    shared = in_frame(np.array([xyz]), np.eye(4), grid['enu_to_ecef'])[0]
    assert shared[:2] == pytest.approx([250, 0], abs=1e-5)


def test_closed_crop_doubles_exact_boundary_and_half_open_fix_removes_it():
    grid = {'enu_to_ecef': np.eye(4).tolist(), 'cells': {
        'centre': {'east_m': [-125, 125], 'north_m': [-125, 125]},
        'east': {'east_m': [125, 375], 'north_m': [-125, 125]}}}
    points = np.array([[124, 0, 0], [125, 0, 0], [126, 0, 0], [0, 125, 0]])
    counts = ownership_counts(points, {'enu_to_ecef': np.eye(4)}, grid)
    assert counts['old_closed_owned_centres'] == 3
    assert counts['old_owned_on_shared_east_125_exact'] == 1
    assert counts['old_owned_within_1m_shared_east_125'] == 2
    assert counts['old_owned_in_neighbour_cell'] == 1
    assert counts['proposed_removed_from_old'] == 2
    assert counts['proposed_centre_owned_centres'] == 1
    assert counts['proposed_doubled_centres'] == 0
    assert cell_mask(points, grid['cells']['east']).tolist() == [False, True, True, False]


def test_audit_uses_shared_frame_for_actual_height():
    grid = read('configs/jax068-grid-neighbour.json')
    # A translated scene can own points east of the common boundary even if x<125 locally.
    geo = np.asarray(grid['enu_to_ecef']).copy()
    geo[:3, 3] += geo[:3, 0] * 0.1
    counts = ownership_counts(np.array([[124.95, 0, 30], [124, 0, 10]]),
                              {'enu_to_ecef': geo}, grid)
    assert counts['old_owned_in_neighbour_cell'] == 1
    assert counts['proposed_centre_owned_centres'] == 1


@pytest.mark.parametrize('offset', [-0.01, 0.01])
def test_neighbour_partition_rejects_any_gap_or_overlap(offset):
    grid = read('configs/jax068-grid-neighbour.json')
    grid['cells']['east']['east_m'][0] += offset
    partition = partition_check(grid)
    assert not partition['passed']
    assert partition['gap_width_m'] == pytest.approx(max(offset, 0))
    assert partition['overlap_width_m'] == pytest.approx(max(-offset, 0))


def test_centre_and_neighbour_tiling_with_actual_grid(tmp_path):
    from plyfile import PlyData, PlyElement
    from pipeline.phase2 import digest
    from pipeline.tiling import plan
    from pipeline.publish import footprint
    grid = read('configs/jax068-grid-neighbour.json')
    fields = list('xyz') + [f'f_dc_{i}' for i in range(3)] + ['opacity']
    fields += [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)]
    data = np.zeros(4, dtype=[(n, '<f4') for n in fields])
    data['x'] = [124, 126, -100, 300]
    data['z'] = 30
    data['rot_0'] = 1
    export = tmp_path / 'export'
    export.mkdir()
    ply = export / 'fused.ply'
    PlyData([PlyElement.describe(data, 'vertex')], text=False).write(ply)
    geo = {'model_to_enu': np.eye(4).tolist(), 'enu_to_ecef': grid['enu_to_ecef'],
           'anchor': grid['anchor']}
    (export / 'transforms_train.json').write_text(json.dumps({'frames': [
        {'transform_matrix': np.eye(4).tolist()}]}))
    (export / 'export-manifest.json').write_text(json.dumps({
        'status': 'exported', 'camera_scene': str(export),
        'fused_ply': {'path': str(ply), 'sha256': digest(ply)}, 'splat_count': 4,
        'georeference': geo, 'round_trip': {'gsd_m': 0.3}, 'tileset': {'rules': []}}))
    profile = {'version': 1, 'block_half_extent_m': 125, 'leaf_max_splats': 64,
               'extent_sigma': 3, 'max_opacity': 0.99}
    rings = []
    for name in ('centre', 'east'):
        result = plan(export, tmp_path / name, profile, grid, name)
        assert result['crop']['ownership_cell'] == name
        assert result['crop']['owned_splats'] == 2
        rings.append(footprint(result)[0]['coordinates'][0])
    assert rings[0][1] == pytest.approx(rings[1][0], abs=1e-10)
    assert rings[0][2] == pytest.approx(rings[1][3], abs=1e-10)


@pytest.fixture
def stage_root(tmp_path):
    (tmp_path / 'artifacts/reference/3dep-jax068-east').mkdir(parents=True)
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'scripts').mkdir()
    return tmp_path


def qualify(root, accepted=True):
    (root / 'artifacts/reference/3dep-jax068-east/qualification.json').write_text(
        json.dumps({'accepted': accepted}))


@pytest.mark.parametrize('accepted', [False, None, 1, 'true', {}, []])
def test_stage_b_rejects_anything_except_boolean_true(stage_root, accepted):
    qualify(stage_root, accepted)
    with pytest.raises(ValueError, match='not qualified'):
        require_stage_b(stage_root)


def test_stage_b_requires_qualification_file(stage_root):
    with pytest.raises(ValueError, match='Missing qualified answer key'):
        require_stage_b(stage_root)


def test_stage_b_requires_decision_even_if_qualified(stage_root):
    qualify(stage_root)
    with pytest.raises(ValueError, match='Missing pre-registered decision'):
        require_stage_b(stage_root)


def test_stage_b_requires_future_checker_and_truth(stage_root):
    qualify(stage_root)
    (stage_root / 'configs/neighbour-decision.json').write_text('{}')
    with pytest.raises(ValueError, match='step 5 checker'):
        require_stage_b(stage_root)
    (stage_root / 'scripts/neighbour-seam-check.py').write_text('# fixture')
    with pytest.raises(ValueError, match='truth payload'):
        require_stage_b(stage_root)
    stable = stage_root / 'artifacts/reference/3dep-jax068-east/stable'
    stable.mkdir()
    for name in ('JAX_068_DSM.tif', 'JAX_068_DSM.txt', 'JAX_068_CLS.tif'):
        (stable / name).write_text('fixture')
    require_stage_b(stage_root)


def test_stage_b_stops_below_disk_limit(stage_root, monkeypatch):
    from collections import namedtuple
    usage = namedtuple('usage', 'total used free')
    monkeypatch.setattr('pipeline.neighbour_preparation.shutil.disk_usage',
                        lambda _: usage(100 * 1024**3, 81 * 1024**3, 19 * 1024**3))
    with pytest.raises(ValueError, match='Less than 20 GiB'):
        require_stage_b(stage_root)


@pytest.mark.parametrize('qualified,decision', [(False, False), (True, False)])
def test_shell_refuses_before_environment_or_output_writes(stage_root, qualified, decision):
    qualify(stage_root, qualified)
    if decision:
        (stage_root / 'configs/neighbour-decision.json').write_text('{}')
    (stage_root / 'pipeline').symlink_to(ROOT / 'pipeline', target_is_directory=True)
    binary = stage_root / '.venv-imagery/bin'
    binary.mkdir(parents=True)
    (binary / 'python').symlink_to(sys.executable)
    (stage_root / 'scripts/stage1-env.sh').write_text('touch ENVIRONMENT_WAS_SOURCED\n')
    result = subprocess.run(['bash', str(ROOT / 'scripts/neighbour-stage-b.sh')],
                            cwd=stage_root, capture_output=True, text=True)
    assert result.returncode == 2
    assert ('not qualified' if not qualified else 'Missing pre-registered decision') in result.stderr
    assert not (stage_root / 'ENVIRONMENT_WAS_SOURCED').exists()
    assert not (stage_root / 'artifacts/stage1').exists()


@pytest.mark.parametrize('passed,holdout', [(True, '104001000A22A000'),
                                         (False, '104001000A22A000'), (True, 'wrong')])
def test_publication_evidence_requires_pass_and_preserves_measured_values(tmp_path, monkeypatch, passed, holdout):
    collect = runpy.run_path(str(ROOT / 'scripts/neighbour-evidence.py'))['collect']
    monkeypatch.chdir(tmp_path)
    def write(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    run = 'artifacts/stage1/eastn-2015-19v-recipe-01'
    tile = 'artifacts/tiles/eastn-2015-19v-recipe-01-r1'
    stats = {'mae_m': 0.42, 'rmse_m': 0.61, 'completeness': 0.8}
    write('artifacts/validation/eastn-2015-19v-recipe-01-lidar/dsm-validation.json',
          {'registered': {k: stats for k in ('all_labelled', 'ground', 'building', 'vegetation')},
           'registration': {'splats_offset_from_lidar_m': {'horizontal': 0.2}}})
    write(f'{run}/evaluation/evaluation-report.json', {'records': {'final': [
        {'role': 'holdout', 'catalog_id': holdout, 'psnr_db': 18.5}]}})
    write('artifacts/validation/neighbour-seam-check.json', {'decision': {'passed': passed}})
    write('artifacts/imagery/eastn-19v/selection.json', {'holdout': [{'catalog_id': '104001000A22A000'}]})
    write(f'{tile}/tiles-report.json', {'tiles': [1, 2], 'totals': {'glbBytes': 1234}})
    write(f'{tile}/tile-plan.json', {'crop': {'owned_splats': 4321}})
    (tmp_path / 'training').mkdir()
    if not passed or holdout == 'wrong':
        with pytest.raises(ValueError):
            collect()
        assert not Path('training/neighbour-eastn-result.json').exists()
    else:
        collect()
        result = json.loads(Path('training/neighbour-eastn-result.json').read_text())
        assert result['quality']['registered_lidar_mae_all_m'] == 0.42
        assert result['quality']['holdout_psnr_db'] == 18.5
        assert result['quality']['splats'] == 4321
        assert len(result['inputs_sha256']) == 5

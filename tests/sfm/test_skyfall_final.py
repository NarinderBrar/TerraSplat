import json

import numpy as np
import pytest

from pipeline.skyfall_final import export
from test_skyfall_scenes import registration

plyfile = pytest.importorskip('plyfile')


def finished_model(path):
    names = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
             'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    rows = [(0, 0, 20, 0, 0, 0, 3, -1, -1, -1, 1, 0, 0, 0),
            (10, 5, 30, 0, 0, 0, -3, -1, -1, -1, 1, 0, 0, 0)]
    data = np.array(rows, dtype=[(n, 'f4') for n in names])
    plyfile.PlyData([plyfile.PlyElement.describe(data, 'vertex')]).write(str(path))


@pytest.fixture
def scene(tmp_path):
    source = tmp_path / 'JAX_000'
    source.mkdir()
    (source / 'transforms_train.json').write_text(json.dumps({'frames': []}))
    (source / 'points3D.txt').write_text('# id x y z\n1 -100 -100 0\n2 100 100 40\n')
    path = tmp_path / 'registration.json'
    path.write_text(json.dumps(registration() | {'scene': 'JAX_000'}))
    return {'scene': 'JAX_000', 'source': str(source), 'registration': str(path)}


def test_finished_model_export_is_placed_without_a_training_run(scene, tmp_path):
    finished_model(tmp_path / 'model.ply')
    manifest = export(scene, tmp_path / 'model.ply', tmp_path / 'placed')
    assert manifest['status'] == 'exported' and manifest['run_directory'] is None
    assert manifest['camera_scene'] == scene['source']
    assert manifest['splat_count'] == 2 and manifest['sh_degree'] == 0
    assert manifest['round_trip']['passed']
    assert json.loads((tmp_path / 'placed/export-manifest.json').read_text()) == manifest
    # Only the opaque splat feeds the display ground; it sits on TerraScope's groundZ.
    surface = np.load(tmp_path / 'placed/surface-points.npz')['enu']
    np.testing.assert_allclose(surface, [[0, 0, 0]], atol=1e-9)
    with pytest.raises(ValueError, match='new placed export'):
        export(scene, tmp_path / 'model.ply', tmp_path / 'placed')

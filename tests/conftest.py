"""Tiny offline tiled run shared by publisher and API integration tests."""

import json

import pytest


@pytest.fixture
def tiled_release(tmp_path):
    import numpy as np

    from pipeline.publish import ROOT, sha256

    source = tmp_path / 'tiles'
    (source / 'tiles').mkdir(parents=True)
    (source / 'tiles/r.glb').write_bytes(b'glTF' + bytes(range(64)))
    # Equator / prime meridian, ENU basis in ECEF; anchor height 0.
    transform = np.array([[0, 0, 1, 6378137], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    plan = {'root_transform': transform.flatten(order='F').tolist(),
            'enu_anchor': {'latitude': 0, 'longitude': 0, 'ellipsoidal_height_m': 0},
            'crop': {'half_extent_m': 125, 'owned_z_range_m': [0, 40]}, 'tiles': [{'id': 'r'}],
            'export_manifest_sha256': 'a' * 64, 'fused_ply_sha256': 'b' * 64}
    tileset = {'asset': {'version': '1.1'}, 'geometricError': 400,
               'root': {'transform': plan['root_transform'], 'boundingVolume': {
                   'box': [0, 0, 20, 125, 0, 0, 0, 125, 0, 0, 0, 20]},
                   'geometricError': 1, 'content': {'uri': 'tiles/r.glb'}}}
    for name, value in [('tile-plan.json', plan), ('tileset.json', tileset)]:
        (source / name).write_text(json.dumps(value))
    report = {'planSha256': sha256(source / 'tile-plan.json'), 'tilesetSha256': sha256(source / 'tileset.json'),
              'tiles': {'r': {'glbSha256': sha256(source / 'tiles/r.glb')}}}
    (source / 'tiles-report.json').write_text(json.dumps(report))
    config = json.loads((ROOT / 'configs/releases/jax068-2015-3v.json').read_text())
    config['tiles_dir'] = str(source)
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config))
    return config_path, tmp_path / 'releases', source

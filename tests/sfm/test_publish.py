import copy
import json
import math

import numpy as np
import pytest

from pipeline.publish import (
    activate, catalog, ecef_to_geodetic, footprint, ground_height, publish, read, sha256, validate_document,
    verify_release,
)


def test_publish_immutable_and_hashes(tiled_release):
    config, releases, _ = tiled_release
    m = publish(config, releases)
    verify_release(releases / m['release_id'], m)
    assert m['tile_count'] == 1
    assert m['historical'] is True
    assert m['quality']['registered_lidar_mae_all_m'] == 3.317
    assert m['quality']['holdout_psnr_db'] == pytest.approx(17.572909692)
    assert catalog(releases)['datasets'][0]['active_release_id'] is None
    before = sha256(releases / m['release_id'] / 'release.json')
    with pytest.raises(ValueError, match='Immutable'):
        publish(config, releases, True)
    assert sha256(releases / m['release_id'] / 'release.json') == before
    assert not list((releases / '.staging').iterdir())


@pytest.mark.parametrize('name', ['tiles/r.glb', 'tileset.json', 'tile-plan.json'])
@pytest.mark.parametrize('missing', [False, True])
def test_reject_changed_or_missing(tiled_release, name, missing):
    config, releases, source = tiled_release
    if missing:
        (source / name).unlink()
    else:
        (source / name).write_bytes(b'changed')
    with pytest.raises(ValueError, match='Hash mismatch|Missing'):
        publish(config, releases)
    assert not (releases / read(config)['release_id']).exists()
    assert not (releases / 'catalog.json').exists()


def test_activation_rollback_index(tiled_release):
    config, releases, _ = tiled_release
    first = publish(config, releases, True)
    cfg = read(config)
    cfg['release_id'] = 'jax068-2015-5v-r1'
    config.write_text(json.dumps(cfg))
    second = publish(config, releases, True)
    assert catalog(releases)['datasets'][0]['active_release_id'] == second['release_id']
    activate(releases, first['release_id'])
    d = catalog(releases)['datasets'][0]
    assert len(d['releases']) == 2
    assert d['active_release_id'] == first['release_id']
    index = read(releases / 'index/jacksonville/tileset.json')
    assert index['asset']['version'] == '1.1'
    root = index['root']
    assert root['refine'] == 'ADD'
    assert len(root['children']) == 1
    child = root['children'][0]
    assert 'transform' not in child
    assert child['content']['uri'] == f'../../{first["release_id"]}/tileset.json'
    assert root['geometricError'] > child['geometricError'] == 400
    # Expected ECEF bounds from the known equatorial ENU box.
    box = child['boundingVolume']['box']
    assert box == pytest.approx([6378157, 0, 0, 20, 0, 0, 0, 125, 0, 0, 0, 125])
    assert root['boundingVolume']['box'] == pytest.approx(box)
    # The outer index repeats the releases' splat declarations for viewers that read only the outermost tileset.
    assert index['extensionsRequired'] == ['3DTILES_content_gltf']
    assert index['extensions']['3DTILES_content_gltf']['extensionsRequired'] == [
        'KHR_gaussian_splatting', 'KHR_gaussian_splatting_compression_spz_2']
    with pytest.raises(ValueError, match='Unknown'):
        activate(releases, 'unknown')
    (releases / first['release_id'] / 'tiles/r.glb').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Hash mismatch'):
        activate(releases, first['release_id'])


@pytest.mark.parametrize('lon,lat,height', [(0, 0, 0), (-81.66, 30.35, -70), (120, -73, 9000),
                                           (0, 90, 12), (0, -90, -20), (179.9, 0.001, 400000)])
def test_ecef_mm_accuracy(lon, lat, height):
    a, e2 = 6378137.0, 6.6943799901413165e-3
    phi, lam = math.radians(lat), math.radians(lon)
    n = a / math.sqrt(1 - e2 * math.sin(phi) ** 2)
    ecef = ((n + height) * math.cos(phi) * math.cos(lam),
            (n + height) * math.cos(phi) * math.sin(lam), (n * (1 - e2) + height) * math.sin(phi))
    actual = ecef_to_geodetic(*ecef)
    assert actual[:2] == pytest.approx((lon, lat), abs=1e-10)
    assert actual[2] == pytest.approx(height, abs=0.001)


def test_owned_square_footprint(tiled_release):
    _, _, source = tiled_release
    polygon, bbox = footprint(read(source / 'tile-plan.json'))
    # Independently computed WGS84 equatorial tangent-square corners (PROJ).
    ring = np.array(polygon['coordinates'][0])
    expected = [[-0.001122894104, -0.001130461846], [0.001122894104, -0.001130461846],
                [0.001122894104, 0.001130461846], [-0.001122894104, 0.001130461846],
                [-0.001122894104, -0.001130461846]]
    assert ring == pytest.approx(np.array(expected), abs=1e-8)
    assert bbox[:4] == pytest.approx([-0.001122894104, -0.001130461846, 0.001122894104, 0.001130461846], abs=1e-8)
    assert bbox[4] == pytest.approx(0, abs=0.001)
    assert 40 < bbox[5] < 40.01


def test_schema_rejects_invalid_manifest(tiled_release):
    config, releases, _ = tiled_release
    m = publish(config, releases)
    for key, value in [('historical', False), ('bbox', [0]), ('release_id', '../escape')]:
        bad = copy.deepcopy(m)
        bad[key] = value
        with pytest.raises(ValueError):
            validate_document(bad, 'release-manifest')


def test_index_multiple_datasets_and_regions(tiled_release):
    config, releases, source = tiled_release
    publish(config, releases, True)
    cfg = read(config)
    cfg.update(dataset_id='adjacent', release_id='adjacent-r1')
    config.write_text(json.dumps(cfg))
    publish(config, releases, True)
    cfg.update(dataset_id='other-city', release_id='other-city-r1', region='other')
    config.write_text(json.dumps(cfg))
    publish(config, releases, True)
    root = read(releases / 'index/jacksonville/tileset.json')['root']
    assert len(root['children']) == 2
    assert {c['content']['uri'] for c in root['children']} == {
        '../../jax068-2015-3v-r1/tileset.json', '../../adjacent-r1/tileset.json'}
    assert len(read(releases / 'index/other/tileset.json')['root']['children']) == 1
    # A catalogue-pinned manifest hash detects changes even to unhashed metadata.
    path = releases / 'adjacent-r1/release.json'
    m = read(path)
    m['title'] = 'Changed'
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match='Manifest hash mismatch'):
        activate(releases, 'adjacent-r1')


def test_failed_publication_cleans_staging(tiled_release):
    config, releases, _ = tiled_release
    cfg = read(config)
    cfg['quality_sources']['completeness']['pointer'] = '/does-not-exist'
    config.write_text(json.dumps(cfg))
    with pytest.raises(KeyError):
        publish(config, releases)
    assert not (releases / cfg['release_id']).exists()
    assert not list((releases / '.staging').iterdir())


def ground_points(path, ground=2.0, roofs=12.0, half=125):
    rng = np.random.default_rng(0)
    xy = rng.uniform(-half, half, size=(40000, 2))
    z = np.where(rng.uniform(size=len(xy)) < 0.35, roofs, ground) + rng.normal(0, 0.05, len(xy))
    outside = np.array([[half + 50, 0, -30.0]])  # floaters outside the owned crop are ignored
    np.savez_compressed(path, enu=np.vstack([np.column_stack([xy, z]), outside]))
    return path


def test_ground_height_is_the_dominant_surface_level(tmp_path):
    plan = {'crop': {'half_extent_m': 125}, 'enu_anchor': {'ellipsoidal_height_m': -70}}
    ground = ground_height(ground_points(tmp_path / 'points.npz'), plan)
    assert ground['ellipsoidal_height_m'] == pytest.approx(-68.0, abs=0.05)
    assert len(ground['source_sha256']) == 64


def test_publish_records_ground_only_when_configured(tiled_release):
    config, releases, source = tiled_release
    assert 'ground' not in publish(config, releases)
    cfg = read(config)
    cfg['release_id'] = 'jax068-2015-3v-ground'
    cfg['ground_points'] = str(ground_points(source / 'depth-points.npz'))
    config.write_text(json.dumps(cfg))
    m = publish(config, releases)
    assert m['ground']['ellipsoidal_height_m'] == pytest.approx(2.0, abs=0.05)
    validate_document(m, 'release-manifest')


def test_pipeline_quality_cannot_be_replaced_with_unmeasured_values(tiled_release):
    config, releases, _ = tiled_release
    cfg = read(config)
    del cfg['quality_sources']['registered_lidar_mae_all_m']
    cfg['quality_values'] = {'registered_lidar_mae_all_m': None}
    config.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='must be measured'):
        publish(config, releases)

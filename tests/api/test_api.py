import json

from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator
import pytest

from pipeline.publish import ROOT, activate, publish, read
from services.api.main import create_app


@pytest.fixture
def client(tiled_release):
    config, releases, _ = tiled_release
    publish(config, releases, True)
    cfg = read(config)
    cfg['release_id'] = 'jax068-2015-5v-r1'
    config.write_text(json.dumps(cfg))
    publish(config, releases, True)
    with TestClient(create_app(releases)) as session:
        yield session, releases


def test_endpoints_and_schema(client):
    c, releases = client
    assert c.get('/api/health').json() == {'status': 'ok'}
    ds = c.get('/api/datasets').json()
    assert ds[0]['active_release_id'] == 'jax068-2015-5v-r1'
    assert ds[0]['active_release']['tileset_url'] == '/releases/jax068-2015-5v-r1/tileset.json'
    detail = c.get('/api/datasets/jax068-2015').json()
    assert len(detail['releases']) == 2
    manifest = c.get('/api/releases/jax068-2015-5v-r1').json()
    quality = c.get('/api/releases/jax068-2015-5v-r1/quality').json()
    assert quality['quality'] == manifest['quality']
    assert 'unknown' in quality['limitations']
    for evidence in quality['evidence']:
        assert c.get(evidence['url']).status_code == 200
        assert len(evidence['sha256']) == 64
    footprints = c.get('/api/footprints').json()
    feature = footprints['features'][0]
    assert feature['geometry'] == manifest['footprint']
    assert feature['properties']['release_id'] == manifest['release_id']
    assert feature['properties']['capture_date_start'].startswith('2014-10-05')
    assert feature['properties']['headline_mae_m'] == pytest.approx(1.3587, abs=1e-4)
    assert feature['properties']['ground_height_m'] is None  # fixture release has no ground_points
    index = c.get('/api/index/jacksonville').json()
    assert c.get(index['tileset_url']).json()['asset']['version'] == '1.1'
    assert c.get('/openapi.json').status_code == 200
    for name, document in [('release-manifest', manifest), ('catalog', read(releases / 'catalog.json'))]:
        schema = read(ROOT / f'schemas/{name}.schema.json')
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(document)
    activate(releases, 'jax068-2015-3v-r1')
    assert c.get('/api/datasets').json()[0]['active_release_id'] == 'jax068-2015-3v-r1'
    assert c.get('/api/footprints').json()['features'][0]['properties']['release_id'] == 'jax068-2015-3v-r1'


@pytest.mark.parametrize('url', ['/api/datasets/unknown', '/api/releases/unknown', '/api/releases/unknown/quality',
                                 '/api/index/unknown', '/releases/unknown/tileset.json'])
def test_unknown(client, url):
    assert client[0].get(url).status_code == 404


@pytest.mark.parametrize('url', ['/api/datasets/Bad_ID', '/api/releases/Bad_ID', '/api/releases/Bad_ID/quality',
                                 '/api/index/Bad_ID'])
def test_invalid_ids(client, url):
    assert client[0].get(url).status_code == 422


def test_static_cors_cache_range_and_traversal(client):
    c, releases = client
    prefix = '/releases/jax068-2015-5v-r1'
    for name, content_type in [('tiles/r.glb', 'model/gltf-binary'), ('release.json', 'application/json'),
                               ('tileset.json', 'application/json')]:
        response = c.get(f'{prefix}/{name}', headers={'Origin': 'http://127.0.0.1:5173'})
        assert response.status_code == 200
        assert response.headers['content-type'] == content_type
        assert 'immutable' in response.headers['cache-control']
        assert response.headers['access-control-allow-origin'] == 'http://127.0.0.1:5173'
        conditional = c.get(f'{prefix}/{name}', headers={'If-None-Match': response.headers['etag']})
        assert conditional.status_code == 304
        assert 'immutable' in conditional.headers['cache-control']
    for name in ['catalog.json', 'index/jacksonville/tileset.json']:
        assert c.get('/releases/' + name).headers['cache-control'] == 'public, max-age=60'
    response = c.get(f'{prefix}/tiles/r.glb', headers={'Range': 'bytes=0-3'})
    assert response.status_code == 206
    assert response.content == b'glTF'
    assert response.headers['content-range'].startswith('bytes 0-3/')
    assert c.head(f'{prefix}/tiles/r.glb').content == b''
    assert 'access-control-allow-origin' not in c.get('/api/health', headers={'Origin': 'https://bad.example'}).headers
    for origin in ['http://127.0.0.1:5173', 'http://127.0.0.1:4173']:
        assert c.get('/api/datasets', headers={'Origin': origin}).headers['access-control-allow-origin'] == origin
    (releases / '.staging/secret').mkdir(parents=True)
    (releases / '.staging/secret/private.json').write_text('{}')
    for url in ['/releases/', prefix + '/', prefix + '/tiles/', '/releases/.publish.lock',
                '/releases/.staging/secret/private.json', '/releases/%2e%2e/configs/releases/jax068-2015-3v.json',
                prefix + '/%2e%2e/%2e%2e/catalog.json', prefix + '/%2e%2e/%2e%2e/secret',
                prefix + '/tiles/%2e%2e/%2e%2e/%2e%2e/secret']:
        assert c.get(url).status_code == 404, url
    assert c.post('/api/health').status_code == 405


def test_empty_and_config(tmp_path, monkeypatch):
    monkeypatch.setenv('TERRASPLAT_RELEASES_DIR', str(tmp_path))
    monkeypatch.setenv('TERRASPLAT_CORS_ORIGINS', 'https://viewer.example')
    c = TestClient(create_app())
    assert c.get('/api/datasets').json() == []
    assert c.get('/api/footprints').json()['features'] == []
    assert c.get('/api/health', headers={'Origin': 'https://viewer.example'}).headers[
        'access-control-allow-origin'] == 'https://viewer.example'


def test_skyfall_quality_can_be_unmeasured_without_claiming_lidar_accuracy(tiled_release):
    config, releases, _ = tiled_release
    cfg = read(config)
    cfg.update(dataset_id='skyfall-nyc-004', release_id='skyfall-nyc-004-stage1-r1', region='new-york',
               source_label="TerraSplat Stage 1 · authors' prepared Skyfall data",
               accuracy_label='TerraScope visual registration; vertical approximate')
    unknown = {key: None for key in cfg['quality_sources'] if key.startswith('registered_lidar_')
               or key in ('horizontal_offset_m', 'completeness')}
    cfg['quality_sources'] = {key: value for key, value in cfg['quality_sources'].items() if key not in unknown}
    cfg['quality_values'] = unknown
    config.write_text(json.dumps(cfg))
    publish(config, releases, True)
    with TestClient(create_app(releases)) as c:
        m = c.get('/api/releases/skyfall-nyc-004-stage1-r1').json()
        assert m['quality']['registered_lidar_mae_all_m'] is None
        assert m['quality']['holdout_psnr_db'] > 0
        feature = c.get('/api/footprints').json()['features'][0]
        assert feature['properties']['headline_mae_m'] is None
        assert feature['properties']['source_label'] == cfg['source_label']
        assert feature['properties']['accuracy_label'] == cfg['accuracy_label']

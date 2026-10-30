import json
from pathlib import Path

import pytest

from pipeline.imagery_check import check, read_images

ROOT = Path(__file__).resolve().parents[2]


def image(index, azimuth, elevation=70, day=1, **changes):
    return {'id': f'img{index}', 'capture_time': f'2026-05-{day:02d}T16:00:{index:02d}Z',
            'sat_azimuth_deg': azimuth, 'sat_elevation_deg': elevation, 'sun_elevation_deg': 60,
            'gsd_m': 0.31, 'cloud_cover': 0.02, 'product_level': 'L1B', 'rpc': True,
            'bands': ['pan', 'msi'], 'covers_aoi': True} | changes


def run(tmp_path, images, name='images.json'):
    path = tmp_path / name
    path.write_text(json.dumps(images))
    return check(read_images(path))


def test_twenty_same_day_views_from_all_sides_match_the_measured_nineteen(tmp_path):
    # Several passes in one day: azimuths every 18 degrees, elevations alternating 60/80.
    images = [image(k, 18 * k, 60 if k % 2 else 80) for k in range(20)]
    report = run(tmp_path, images)
    assert report['verdict'] == 'expected'
    assert report['measured_reference']['mae_m'] == 1.36
    assert report['training_geometry']['views'] == 19 and report['holdout'] is not None
    assert report['training_geometry']['distinct_days'] == 1 and report['warnings'] == []


def test_single_pass_is_flagged_and_not_given_the_best_tier(tmp_path):
    # One along-track pass: forward and backward views share one azimuth plane.
    images = [image(k, 10 if k < 10 else 190, 60 + 2 * (k % 10)) for k in range(20)]
    report = run(tmp_path, images)
    assert report['verdict'] == 'unmeasured'
    assert any('one viewing plane' in w for w in report['warnings'])
    one_side = [image(k, 4 * k, 60 + k) for k in range(20)]
    report = run(tmp_path, one_side)
    assert report['verdict'] == 'unmeasured'
    assert any('one side' in w for w in report['warnings'])


def test_repeated_azimuths_are_not_a_full_circle_gap(tmp_path):
    report = run(tmp_path, [image(k, a) for k, a in enumerate([0, 0, 90, 180, 270])])
    assert report['training_geometry']['max_azimuth_gap_deg'] == 90


def test_unusable_products_are_rejected_with_reasons(tmp_path):
    images = [image(0, 0), image(1, 120, product_level='ORTHO'), image(2, 240, rpc=False),
              image(3, 60, sat_elevation_deg=45), image(4, 300, cloud_cover=0.4), image(5, 180, gsd_m=0.7)]
    report = run(tmp_path, images)
    assert report['usable'] == ['img0'] and report['verdict'] == 'reject'
    reasons = ' '.join(sum(report['rejected'].values(), []))
    for text in ('ORTHO', 'no RPC', 'off-nadir 45.0', 'cloud cover 40%', 'GSD 0.70'):
        assert text in reasons


def test_csv_input_and_span_warning(tmp_path):
    rows = [image(k, 90 * k, 70, day=1 + 5 * k) for k in range(5)]
    header = list(rows[0])
    lines = [','.join(header)] + [','.join('pan+msi' if h == 'bands' else str(r[h]) for h in header) for r in rows]
    (tmp_path / 'images.csv').write_text('\n'.join(lines) + '\n')
    report = check(read_images(tmp_path / 'images.csv'))
    assert report['training_geometry']['views'] == 4
    assert any('span' in w for w in report['warnings'])


def test_missing_fields_are_reported(tmp_path):
    with pytest.raises(ValueError, match='missing rpc'):
        run(tmp_path, [{k: v for k, v in image(0, 0).items() if k != 'rpc'}])


def test_measured_nineteen_view_set_reproduces_its_own_tier(tmp_path):
    selection = ROOT / 'artifacts/imagery/pilot-19v/selection.json'
    if not selection.is_file():
        pytest.skip('19-view selection not available')
    value = json.loads(selection.read_text())
    images = [{'id': i['catalog_id'], 'capture_time': i['capture_time'], 'sat_azimuth_deg': i['sat_azimuth_deg'],
               'sat_elevation_deg': i['sat_elevation_deg'], 'sun_elevation_deg': i['sun_elevation_deg'],
               'gsd_m': i['gsd_m'], 'cloud_cover': i['cloud_cover'], 'product_level': 'L1B', 'rpc': True,
               'bands': ['pan', 'msi'], 'covers_aoi': True} for i in value['train'] + value['holdout']]
    path = tmp_path / 'images.json'
    path.write_text(json.dumps(images))
    report = check(read_images(path), value['holdout'][0]['catalog_id'])
    assert report['measured_reference']['name'] == '19-view recipe'

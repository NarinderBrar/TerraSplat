import hashlib
import json

import numpy as np
import pytest

from pipeline.publish import copy_segments
from pipeline.segments import majority


def test_majority_prefers_known_classes_and_keeps_all_unknown():
    labels = np.array([[0, 0, 0, 3, 3, 4], [0, 0, 0, 0, 0, 0], [4, 4, 3, 0, 0, 0]])
    assert majority(labels).tolist() == [3, 0, 4]


def write_layer(directory, tiles):
    directory.mkdir()
    entries = {}
    for tile_id, values in tiles.items():
        data = bytes(values)
        (directory / f'{tile_id}.cls').write_bytes(data)
        entries[tile_id] = {'splats': len(values), 'leaf': True, 'sha256': hashlib.sha256(data).hexdigest()}
    (directory / 'segments.json').write_text(json.dumps({'label': 'test layer', 'tiles': entries}))


def test_copy_segments_requires_every_tile_splat_for_splat(tmp_path):
    plan = {'tiles': [{'id': 'r', 'splats': 2}, {'id': 'r0', 'splats': 1}]}
    write_layer(tmp_path / 'good', {'r': [3, 4], 'r0': [1]})
    (tmp_path / 'release').mkdir()
    assert copy_segments(tmp_path / 'good', plan, tmp_path / 'release') == {'path': 'segments/segments.json', 'label': 'test layer'}
    assert (tmp_path / 'release/segments/r.cls').read_bytes() == bytes([3, 4])

    write_layer(tmp_path / 'short', {'r': [3, 4]})
    (tmp_path / 'release2').mkdir()
    with pytest.raises(ValueError, match='does not match the tile plan'):
        copy_segments(tmp_path / 'short', plan, tmp_path / 'release2')

    (tmp_path / 'good/r.cls').write_bytes(bytes([3, 5]))
    (tmp_path / 'release3').mkdir()
    with pytest.raises(ValueError, match='changed'):
        copy_segments(tmp_path / 'good', plan, tmp_path / 'release3')


def test_withdraw_moves_an_inactive_release_out_of_the_catalogue(tmp_path):
    from pipeline.publish import catalog, save_catalog, withdraw
    entry = lambda r: {'release_id': r, 'title': r, 'manifest_path': f'{r}/release.json', 'manifest_sha256': '0' * 64}
    for r in ('a-r1', 'a-r2'):
        (tmp_path / r).mkdir()
        (tmp_path / r / 'release.json').write_text('{}')
    save_catalog(tmp_path, {'schema_version': 1, 'datasets': [{'dataset_id': 'a', 'title': 'A', 'region': 'x',
                                                               'active_release_id': None, 'releases': [entry('a-r1'), entry('a-r2')]}]})
    assert withdraw(tmp_path, 'a-r2') == tmp_path / '.withdrawn' / 'a-r2'
    assert [r['release_id'] for r in catalog(tmp_path)['datasets'][0]['releases']] == ['a-r1']
    assert not (tmp_path / 'a-r2').exists() and (tmp_path / '.withdrawn/a-r2/release.json').exists()
    with pytest.raises(ValueError, match='Unknown release'):
        withdraw(tmp_path, 'a-r2')

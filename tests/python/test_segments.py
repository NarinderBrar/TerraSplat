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

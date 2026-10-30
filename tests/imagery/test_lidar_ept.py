import json

import numpy as np

from pipeline import lidar_ept


def test_node_bounds_halve_per_depth():
    low, high = lidar_ept.node_bounds([0, 0, 0, 8, 8, 8], '1-1-0-1')
    np.testing.assert_allclose(low, [4, 0, 4])
    np.testing.assert_allclose(high, [8, 4, 8])


def test_chunk_keys_follow_subtrees_and_skip_empty_or_outside(monkeypatch):
    files = {
        '0-0-0-0': {'0-0-0-0': 10, '1-0-0-0': -1, '1-1-1-0': 5, '1-0-1-0': 0},
        '1-0-0-0': {'1-0-0-0': 7, '2-0-0-0': 3, '2-1-1-0': 2},
    }
    monkeypatch.setattr(lidar_ept, 'get', lambda url: json.dumps(files[url.rsplit('/', 1)[1][:-5]]).encode())
    # Box inside the lower-left quarter: the root, the 1-0-0-0 subtree and its overlapping children only.
    keys = lidar_ept.chunk_keys('base', [0, 0, 0, 8, 8, 8], (0.5, 0.5, 3.5, 3.5))
    assert keys == ['0-0-0-0', '1-0-0-0', '2-0-0-0', '2-1-1-0']

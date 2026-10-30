import numpy as np
import pytest

from pipeline.semantics import CODES
from pipeline.semantics_clip import classify_clip, sample_probabilities, segment_probabilities, visible_pixels

NAMES = ['building', 'tree', 'grass', 'road', 'pavement', 'soil', 'water', 'car', 'shadow']
WEIGHTS = {'default': 0.45, 's': 0.30, 'm': 0.20, 'l': 0.05}


def test_segment_softmax_prefers_the_nearest_prototype():
    prototypes = np.eye(3, 4, dtype=np.float32)
    probabilities = segment_probabilities(np.array([[2.0, 0.1, 0, 0], [0, 0, 0.5, 0]], np.float32), prototypes)
    assert probabilities.sum(axis=1) == pytest.approx([1, 1])
    assert probabilities.argmax(axis=1).tolist() == [0, 2]


def test_levels_without_a_segment_are_dropped_and_renormalised():
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0]], np.float32)
    seg_maps = np.array([[[0, -1, -1]], [[1, 1, -1]], [[-1, -1, -1]], [[-1, -1, -1]]])  # levels x 1 x 3
    result = sample_probabilities(seg_maps, probabilities, np.zeros(3, int), np.arange(3), WEIGHTS)
    assert result[0] == pytest.approx([0.6, 0.4])  # 0.45 vs 0.30
    assert result[1] == pytest.approx([0.0, 1.0])
    assert np.isnan(result[2]).all()


def test_z_buffer_hides_points_behind_nearer_ones_and_respects_coverage():
    u = np.array([1.2, 1.4, 3.5, 3.5, 9.0])
    v = np.array([1.2, 1.3, 1.5, 1.5, 1.0])
    depth = np.array([10.0, 15.0, 10.0, 10.5, 10.0])
    coverage = np.ones((4, 6), np.uint8)
    coverage[1, 3] = 0
    ok, _, _ = visible_pixels(u, v, depth, (4, 6), coverage, tolerance=1.0)
    # second point is 5 units behind the first; points 3/4 are uncovered; the last is out of frame
    assert ok.tolist() == [True, False, False, False, False]


def test_v3_rules_use_height_only_for_raised_versus_flat():
    hag = np.array([[0.5, 5.0, 0.5, 5.0, 0.5, 5.0, 0.5, 5.0]])
    picks = ['tree', 'grass', 'building', 'building', 'pavement', 'pavement', 'car', 'water']
    probabilities = np.full(hag.shape + (len(NAMES),), 0.01, np.float32)
    for i, name in enumerate(picks):
        probabilities[0, i, NAMES.index(name)] = 0.9
    views = np.full(hag.shape, 5)
    views[0, 7] = 1
    classes, clip_only, _, known = classify_clip(probabilities, views, hag, np.ones(hag.shape, bool), NAMES, 2.5, 2)
    c = CODES
    assert classes.tolist() == [[c['vegetation'], c['grass'], c['ground'], c['building'], c['ground'], c['unknown'],
                                 c['road'], c['unknown']]]
    assert clip_only[0, :7].tolist() == [c['vegetation'], c['grass'], c['building'], c['building'], c['ground'],
                                         c['ground'], c['road']]
    assert not known[0, 7]  # one view is not enough


def test_v4_parking_follows_the_flat_surface_rule():
    names = ['building', 'tree', 'grass', 'road', 'parking', 'soil', 'water']
    probabilities = np.full((1, 2, len(names)), 0.01, np.float32)
    probabilities[..., names.index('parking')] = 0.9
    classes, clip_only, _, _ = classify_clip(probabilities, np.full((1, 2), 3), np.array([[0.5, 4.0]]),
                                             np.ones((1, 2), bool), names, 2.5, 2)
    assert classes.tolist() == [[CODES['ground'], CODES['unknown']]]
    assert clip_only.tolist() == [[CODES['ground'], CODES['ground']]]

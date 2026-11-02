import json
import warnings

import numpy as np
import pytest

from pipeline.semantics_landcover import NAMES, fuse_views, view_probabilities


@pytest.mark.parametrize('frame_count', [3, 4])
def test_disk_fusion_matches_full_median_across_chunk_boundary(tmp_path, frame_count):
    scene, probs = tmp_path / 'scene', tmp_path / 'probs'
    (scene / 'masks').mkdir(parents=True)
    probs.mkdir()
    frames = []
    rng = np.random.default_rng(42)
    for index in range(frame_count):
        name = f'frame_{index}'
        frames.append({'file_path': f'{name}.jpg', 'fl_x': 1, 'fl_y': 1, 'cx': 0, 'cy': 0,
                       'transform_matrix': np.eye(4).tolist()})
        probabilities = rng.random((4, 4, len(NAMES)), dtype=np.float32)
        np.save(probs / f'{name}_probs.npy', probabilities)
        mask = np.ones((4, 4), np.uint8)
        mask[1, index % 2] = 0
        np.save(scene / 'masks' / f'{name}.npy', mask)
    transforms = {'frames': frames}  # NYC views infer image dimensions from their probabilities.
    (scene / 'transforms_train.json').write_text(json.dumps(transforms))
    points = np.resize(np.array([[0.2, 1.2, 1], [1.2, 1.2, 1], [8, 8, 1], [0.4, 2.4, 2]]), (16389, 3))
    full = np.stack([view_probabilities(scene, f, transforms, probs, points, 0.1) for f in frames])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        expected = np.nanmedian(full, axis=0)

    median, counts, used = fuse_views(scene, probs, points, 0.1)

    np.testing.assert_array_equal(median, expected)
    np.testing.assert_array_equal(counts, np.isfinite(full[..., 0]).sum(axis=0))
    assert [v['visible_cells'] for v in used] == np.isfinite(full[..., 0]).sum(axis=1).tolist()

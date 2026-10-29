import numpy as np
import pytest

from pipeline.georeference import placement
from pipeline.skyfall_inputs import (
    deskew,
    model_camera,
    normalization,
    project,
    transform_points,
    warp_rgb,
)


@pytest.mark.parametrize("skew", [-80.0, 0.0, 80.0])
def test_full_deskew_canvas_preserves_pixel_centers_and_masks(skew):
    K = np.eye(4)
    K[0, 0] = K[1, 1] = 80
    K[0, 1] = skew
    pixels = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    H, corrected, size = deskew(K, [6, 4])
    warped, mask = warp_rgb(pixels, H, size, np.ones(pixels.shape[:2], dtype=np.uint8))
    assert corrected[0, 1] == 0
    assert mask.sum() == 24
    for row in range(4):
        for col in range(6):
            x, y, _ = H @ [col, row, 1]
            np.testing.assert_array_equal(warped[int(y), int(x)], pixels[row, col])
    if skew == 0:
        assert size == [6, 4]
        np.testing.assert_array_equal(warped, pixels)


def test_subpixel_warp_uses_camera_pixel_centers():
    K = np.eye(4)
    K[0, 0] = K[1, 1] = 100
    K[0, 1] = 25
    pixels = np.broadcast_to(
        np.arange(8, dtype=np.uint8)[None, :, None] * 20, (4, 8, 3)
    ).copy()
    H, _, size = deskew(K, [8, 4])
    warped, mask = warp_rgb(pixels, H, size, np.ones(pixels.shape[:2], dtype=np.uint8))
    # Output x=3, y=2 maps to source x=2.5 at integer row center 2.
    assert H[0, 2] == 1
    assert mask[2, 3] == 1
    np.testing.assert_array_equal(warped[2, 3], [50, 50, 50])


def test_normalization_and_warp_preserve_rays_and_restore_enu():
    xyz = np.array([[-10, -5, 2], [0, 0, 5], [7, 2, 9], [15, 8, 3]], dtype=np.float64)
    norm = normalization(xyz)
    geographic = placement(
        {"longitude": -81.66, "latitude": 30.34, "ellipsoidal_height_m": -70}, norm
    )
    model = transform_points(xyz, geographic["source_to_model"])
    np.testing.assert_allclose(
        transform_points(model, geographic["model_to_enu"]), xyz, atol=1e-12
    )
    K = np.eye(4)
    K[:3, :3] = [[4e6, 70000, -120000], [0, 4e6, 800000], [0, 0, 1]]
    W2C = np.eye(4)
    W2C[2, 3] = 600000
    camera = {
        "K": K.ravel().tolist(),
        "W2C": W2C.ravel().tolist(),
        "img_size": [1200, 900],
    }
    H, corrected_K, size = deskew(K, camera["img_size"])
    corrected, c2w = model_camera(
        camera, corrected_K, size, geographic["source_to_model"], norm["rotation"]
    )
    target = np.column_stack((project(camera, xyz), np.ones(len(xyz)))) @ H.T
    np.testing.assert_allclose(
        project(corrected, model), target[:, :2], rtol=0, atol=1e-8
    )
    np.testing.assert_allclose(c2w[:3, :3].T @ c2w[:3, :3], np.eye(3), atol=1e-12)


def test_degenerate_training_cloud_is_rejected():
    with pytest.raises(ValueError, match="extent"):
        normalization(np.ones((10, 3)))


def test_invalid_source_pixels_are_masked_without_discarding_valid_black_pixels():
    pixels = np.zeros((4, 8, 3), dtype=np.uint8)
    valid = np.ones((4, 8), dtype=np.uint8)
    valid[2, 2] = 0
    H = np.array([[1, -0.25, 1], [0, 1, 0], [0, 0, 1]])
    warped, mask = warp_rgb(pixels, H, [9, 4], valid)
    assert mask[0, 2] == 1
    assert mask[2, 2] == 0
    assert mask[2, 3] == 0
    assert not warped.any()

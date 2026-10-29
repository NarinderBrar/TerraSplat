import numpy as np
import pytest

from pipeline.render_depth import back_project, nadir_camera


def test_nadir_back_projection_recovers_a_ground_plane():
    camera = nadir_camera((10.0, -5.0), 2000.0, 100.0, 400)
    depth = np.full((400, 400), 2000.0 - 3.0)
    points = back_project(depth, camera).reshape(400, 400, 3)
    assert np.allclose(points[..., 2], 3.0)
    # Column grows east, row grows south; pixel size is 200 m / 400 at the plane (x 1997/2000).
    pixel = 200.0 / 400 * 1997 / 2000
    assert points[0, 1, 0] - points[0, 0, 0] == pytest.approx(pixel)
    assert points[1, 0, 1] - points[0, 0, 1] == pytest.approx(-pixel)
    centre = points[199:201, 199:201].reshape(-1, 3).mean(0)
    assert centre[:2] == pytest.approx([10.0, -5.0])


def test_nadir_camera_is_a_proper_rotation_looking_down():
    camera = nadir_camera((0.0, 0.0), 500.0, 50.0, 100)
    rotation = camera["rotation"]
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert rotation[:, 2] == pytest.approx([0, 0, -1])
    assert camera["translation"] == pytest.approx(-rotation.T @ camera["centre"])

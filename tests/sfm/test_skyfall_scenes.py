import math

import numpy as np
import pytest

from pipeline.skyfall_scenes import registration_placement
from pipeline.tiling import covariances, to_enu


def registration(angle=0, px_per_unit=2):
    c, s = math.cos(angle), math.sin(angle)
    return {"tileOrigin": {"z": 18, "x0": 71602, "y0": 107856},
            "modelRaster": {"left": -600, "top": 600, "unitsPerPixel": 1},
            "modelRasterToReferencePixels": [[px_per_unit*c, -px_per_unit*s, 0],
                                              [px_per_unit*s, px_per_unit*c, 0]],
            "georeference": {"groundZ": 20}}


def test_registration_removes_mercator_scale_and_preserves_up():
    geo = registration_placement(registration(math.pi / 6), -17)
    scale = 2 * 2 * math.pi * 6378137 / (256 * 2**18) * math.cos(math.radians(geo['anchor']['latitude']))
    matrix = np.asarray(geo['model_to_enu'])
    assert geo['metres_per_model_unit'] == pytest.approx(scale)
    assert geo['rotation_about_up_deg'] == pytest.approx(-30)
    np.testing.assert_allclose(matrix @ [0, 0, 20, 1], [0, 0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(np.asarray(geo['enu_to_ecef']) @ matrix, geo['model_to_ecef'])
    # The anchor must come from the affine and raster origin, not model bounds.
    point = registration_placement(registration(), -17)['anchor']
    assert point['longitude'] == pytest.approx((71602*256+1200)/(256*2**18)*360-180)


def test_registration_rejects_shear():
    value = registration()
    value['modelRasterToReferencePixels'][0][1] = 0.3
    with pytest.raises(ValueError, match='similarity'):
        registration_placement(value, 0)


def test_up_rotation_rotates_full_covariance():
    matrix = np.array([[0, -2, 0, 3], [2, 0, 0, 4], [0, 0, 2, 5], [0, 0, 0, 1]])
    splats = {'centres': np.array([[1., 2, 3]]), 'log_scales': np.log([[1., 2, 3]]),
              'quaternions': np.array([[1., 0, 0, 0]])}
    result = to_enu(splats, matrix)
    np.testing.assert_allclose(result['centres'], [[-1, 6, 11]])
    old = covariances(np.exp(splats['log_scales']), splats['quaternions'])
    actual = covariances(np.exp(result['log_scales']), result['quaternions'])
    np.testing.assert_allclose(actual, matrix[:3, :3] @ old @ matrix[:3, :3].T, atol=1e-12)


def test_up_rotation_rejects_nonuniform_scale():
    with pytest.raises(ValueError, match='uniform scale'):
        to_enu({}, np.diag([1, 2, 3, 1]))


def test_egm96_sign_when_grid_is_available():
    pytest.importorskip('pyproj')
    from pipeline.skyfall_scenes import ground_ellipsoid

    height, method = ground_ellipsoid(-81.66376, 30.34882, 'JAX')
    assert -19 < height < -16
    assert 'street 12 m' in method

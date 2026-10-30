from pathlib import Path

import laspy
import numpy as np
import pytest
from pyproj import Transformer

from pipeline.lidar_reference import (HeightConversion, qualification, rasterize,
                                      read_points, stable_mask, write_truth)


@pytest.mark.parametrize('extension', ['las', 'laz'])
def test_synthetic_las_conversion_and_highest_non_noise_point(tmp_path, extension):
    lon, lat = -81.66376054127814, 30.34881972282938
    x, y = Transformer.from_crs(4326, 3857, always_xy=True).transform(lon, lat)
    header = laspy.LasHeader(point_format=3, version='1.2')
    header.offsets = [x, y, 0]
    header.scales = [0.001, 0.001, 0.001]
    las = laspy.LasData(header)
    las.x, las.y = np.full(3, x), np.full(3, y)
    las.z, las.classification = np.array([0, 10, 100]), np.array([2, 1, 7])
    path = tmp_path / f'synthetic.{extension}'
    las.write(path)
    converter = HeightConversion(Path.home() / '.local/share/proj/us_noaa_g2018u0.tif')
    points = read_points(path)
    xyz = converter.convert(points)
    # Independently computed anchor: H+N then inverse EPSG:7807 at 2010.0.
    assert xyz[0, 2] == pytest.approx(-29.751054, abs=0.0001)
    assert xyz[1, 2] - xyz[0, 2] == pytest.approx(10, abs=1e-5)
    expected_xy = Transformer.from_crs(4326, 32617, always_xy=True).transform(lon, lat)
    np.testing.assert_allclose(xyz[:, :2], np.tile(expected_xy, (3, 1)), atol=1e-6)
    grid = {'west': xyz[0, 0] - 0.25, 'north': xyz[0, 1] + 0.25, 'resolution': 0.5, 'size': 2}
    dsm, classes, count = rasterize(xyz, points['classification'], grid)
    assert dsm[0, 0] == xyz[1, 2]
    assert classes[0, 0] == 1
    assert count[0, 0] == 2
    assert np.isnan(dsm[1, 1]) and classes[1, 1] == 65


def test_stability_threshold_missing_points_and_unremoved_offset():
    old = np.array([[1., 1.5, 1.50001, np.nan, 1.]])
    new = np.ones_like(old)
    old_count = np.array([[1, 1, 1, 1, 0]])
    result = stable_mask(old, new, old_count, np.ones_like(old_count))
    assert result.tolist() == [[True, True, False, False, False]]
    assert not stable_mask(new + 0.6, new, np.ones_like(old_count), np.ones_like(old_count)).any()


@pytest.mark.parametrize('predicted,accepted', [([0.5, -0.5], True), ([0.51, -0.51], False),
                                               ([0.31, 0.31], False), ([0.3, 0.3], True)])
def test_fixed_qualification_thresholds(predicted, accepted):
    result = qualification(np.array(predicted), np.zeros(2), np.ones(2, bool))
    assert result['accepted'] is accepted


def test_empty_qualification_cannot_pass():
    assert not qualification(np.array([np.nan]), np.zeros(1), np.ones(1, bool))['accepted']


def test_raster_boundaries_and_class_follow_highest_point():
    xyz = np.array([[0.1, 0.9, 3], [0.2, 0.8, 1], [1, 0.9, 10], [0.1, 0, 11]])
    dsm, cls, counts = rasterize(xyz, np.array([6, 2, 6, 6]),
                                 {'west': 0, 'north': 1, 'resolution': 1, 'size': 1})
    assert dsm.tolist() == [[3]]
    assert cls.tolist() == [[6]]
    assert counts.tolist() == [[2]]


def test_truth_output_roundtrips_through_validator(tmp_path):
    from pipeline.dsm_validation import read_truth
    dsm = np.array([[1., np.nan], [3., 4.]])
    cls = np.array([[2, 65], [5, 6]], dtype=np.uint8)
    grid = {'west': 436000., 'north': 3358000., 'resolution': 0.5, 'size': 2}
    write_truth(tmp_path / 'truth', dsm, cls, grid)
    loaded = read_truth(tmp_path / 'truth', 'JAX_068')
    np.testing.assert_array_equal(loaded['dsm'], dsm)
    np.testing.assert_array_equal(loaded['classes'], cls)
    assert loaded['west'] == grid['west']
    assert loaded['north'] == grid['north']
    assert loaded['resolution'] == 0.5


def test_missing_geoid_grid_stops(tmp_path):
    with pytest.raises(FileNotFoundError, match='Offline GEOID18 grid needed'):
        HeightConversion(tmp_path / 'us_noaa_g2018u0.tif')

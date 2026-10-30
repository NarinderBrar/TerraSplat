from datetime import datetime, timezone
import math

import numpy as np
import pytest

from pipeline.semantics import CODES, classify, earth_sun_distance, evaluate, osm_masks, parse_imd, toa_reflectance

IMD = """
firstLineTime = 2015-01-27T16:08:45.999175Z;
meanSunEl = 40.0;
BEGIN_GROUP = BAND_R
\tabsCalFactor = 1.000000e-02;
\teffectiveBandwidth = 5.000000e-02;
END_GROUP = BAND_R
BEGIN_GROUP = BAND_N
\tabsCalFactor = 2.000000e-02;
\teffectiveBandwidth = 1.000000e-01;
END_GROUP = BAND_N
"""


def test_imd_parsing_and_earth_sun_distance():
    imd = parse_imd(IMD)
    assert imd['sun_elevation_deg'] == 40.0
    assert imd['bands']['BAND_N'] == {'absCalFactor': 0.02, 'effectiveBandwidth': 0.1}
    # Perihelion in early January, aphelion in early July.
    assert earth_sun_distance(datetime(2015, 1, 3, tzinfo=timezone.utc)) == pytest.approx(0.9833, abs=2e-4)
    assert earth_sun_distance(datetime(2015, 7, 4, tzinfo=timezone.utc)) == pytest.approx(1.0167, abs=2e-4)


def test_toa_reflectance_formula():
    imd = parse_imd(IMD)
    rho = toa_reflectance(np.array([500]), imd['bands']['BAND_R'], imd, 1535.33)
    d = earth_sun_distance(datetime.fromisoformat('2015-01-27T16:08:45.999175+00:00'))
    expected = math.pi * (500 * 0.01 / 0.05) * d ** 2 / (1535.33 * math.cos(math.radians(50)))
    assert rho[0] == pytest.approx(expected)


def test_rule_priority_and_unmapped_buildings():
    hag = np.array([[0.2, 0.2, 5.0, 5.0, 5.0, 0.2, 1.5, np.nan]])
    ndvi = np.array([[0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
    valid = np.isfinite(hag)
    building = np.array([[0, 0, 1, 0, 0, 0, 0, 0]], bool)
    water = np.array([[1, 1, 0, 0, 0, 0, 0, 0]], bool)
    road = np.array([[0, 0, 0, 0, 1, 1, 0, 0]], bool)
    classes, unmapped = classify(np.nan_to_num(hag, nan=0), ndvi, valid, building, water, road)
    assert classes.tolist() == [[CODES['water'], CODES['water'], CODES['building'], CODES['building'],
                                 CODES['unknown'], CODES['road'], CODES['unknown'], CODES['unknown']]]
    assert unmapped.tolist() == [[False, False, False, True, False, False, False, False]]


def test_vegetation_beats_building_and_ground():
    hag = np.array([[6.0, 0.3]])
    classes, _ = classify(hag, np.array([[0.4, 0.4]]), np.ones((1, 2), bool),
                          np.ones((1, 2), bool), np.zeros((1, 2), bool), np.zeros((1, 2), bool))
    assert classes.tolist() == [[CODES['vegetation'], CODES['vegetation']]]


def test_strict_evaluation_counts_unknown_as_wrong():
    mapping = {'building': 6, 'vegetation': 5, 'water': 9, 'road': 2, 'ground': 2, 'unknown': None}
    ours = np.array([[CODES['building'], CODES['road'], CODES['unknown'], CODES['vegetation']]], np.uint8)
    truth = np.array([[6, 2, 6, 6]])
    result = evaluate(ours, truth, mapping, np.ones(truth.shape, bool))
    assert result['overall_accuracy'] == 0.5
    assert result['unknown_cells'] == 1
    assert result['building']['iou'] == pytest.approx(1 / 3)
    assert result['building']['recall'] == pytest.approx(1 / 3) and result['building']['precision'] == 1.0


def test_osm_polygons_holes_and_road_buffers():
    pyproj = pytest.importorskip('pyproj')
    to_ll = pyproj.Transformer.from_crs(32617, 4326, always_xy=True)
    grid = {'west': 436000.0, 'north': 3357100.0, 'resolution': 1.0}

    def ring(points):
        lon, lat = to_ll.transform([p[0] for p in points], [p[1] for p in points])
        return [{'lon': a, 'lat': b} for a, b in zip(lon, lat)]

    square = lambda x0, y0, x1, y1: ring([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)])
    osm = {'elements': [
        {'type': 'way', 'tags': {'building': 'yes'}, 'geometry': square(436002, 3357092, 436008, 3357098)},
        {'type': 'relation', 'tags': {'natural': 'water'}, 'members': [
            {'role': 'outer', 'geometry': square(436010, 3357080, 436020, 3357090)},
            {'role': 'inner', 'geometry': square(436014, 3357084, 436016, 3357086)}]},
        {'type': 'way', 'tags': {'highway': 'service'}, 'geometry': ring([(436000, 3357070.5), (436030, 3357070.5)])},
    ]}
    building, water, road = osm_masks(osm, grid, {'service': 2.5, 'other': 2}, (40, 40))
    assert building[5, 5] and not building[15, 15]
    assert water[13, 12] and not water[15, 15]  # the inner ring is a hole
    assert road[29, 10] and road[27, 10] and not road[25, 10]


def test_v2_splits_trees_from_grass_and_widens_the_low_band():
    hag = np.array([[3.0, 0.5, 1.6, 1.6]])
    ndvi = np.array([[0.5, 0.5, 0.1, 0.1]])
    road = np.array([[0, 0, 0, 1]], bool)
    thresholds = {'low_hag_m': 2.0, 'tree_hag_m': 2.0}
    classes, _ = classify(hag, ndvi, np.ones((1, 4), bool), np.zeros((1, 4), bool), np.zeros((1, 4), bool), road, thresholds)
    assert classes.tolist() == [[CODES['vegetation'], CODES['grass'], CODES['ground'], CODES['road']]]
    # v1 thresholds leave the 1.6 m cells unknown and call both green cells vegetation.
    v1, _ = classify(hag, ndvi, np.ones((1, 4), bool), np.zeros((1, 4), bool), np.zeros((1, 4), bool), road)
    assert v1.tolist() == [[CODES['vegetation'], CODES['vegetation'], CODES['unknown'], CODES['unknown']]]

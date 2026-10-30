import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.grid_seams import decide, pair_frame, seam_axis

ROOT = Path(__file__).resolve().parents[2]
GRID = json.loads((ROOT / 'configs/jax068-grid-1km.json').read_text())
RULES = json.loads((ROOT / 'configs/grid-1km-decision.json').read_text())


def test_seam_axis_for_vertical_and_horizontal_neighbours():
    assert seam_axis(GRID, 'A2', 'A3') == 0
    assert seam_axis(GRID, 'B2', 'A2') == 1
    with pytest.raises(ValueError):
        seam_axis(GRID, 'A2', 'B3')


def test_horizontal_seam_becomes_vertical_in_the_pair_frame():
    points = np.array([[-100.0, -5.0, 1.0], [-100.0, 5.0, 2.0]])
    swapped, frame = pair_frame(GRID, 'B2', 'A2', points)
    np.testing.assert_array_equal(swapped, [[-5.0, -100.0, 1.0], [5.0, -100.0, 2.0]])
    assert frame['cells']['west']['east_m'][1] == 0 and frame['cells']['east']['east_m'][0] == 0
    # The point south of north=0 lies in the west (south) cell of the pair frame, the other in the east (north) cell.
    w, e = frame['cells']['west'], frame['cells']['east']
    assert w['east_m'][0] <= swapped[0, 0] < w['east_m'][1] and e['east_m'][0] <= swapped[1, 0] < e['east_m'][1]


def test_decide_applies_same_area_baseline_and_residual_steps():
    def cell(mae, coverage=0.26):
        return {'report': {'registered': {'all_labelled': {'mae_m': mae}}, 'coverage': {'block_area_covered_fraction': coverage}}}
    def seam(step_ground, step_all, band=1.0, interior=1.0, doubled=0):
        errors = {side: {'seam': {'pixels': 10, 'mae_m': band}, 'interior': {'pixels': 10, 'mae_m': interior}} for side in ('west', 'east')}
        steps = {k: {'unregistered': {'residual_step_m': v}} for k, v in (('ground', step_ground), ('all', step_all))}
        return {'doubled_centres': doubled, 'metrics': {'lidar_error_by_distance': errors, 'height_continuity': steps}}
    baseline = {'A2': 1.4, 'A3': 1.1}
    rules = RULES['rules']
    good = decide({'A2': cell(1.6), 'A3': cell(1.3)}, {'A2|A3': seam(0.3, 0.2)}, baseline, rules)
    assert good['passed']
    bad = decide({'A2': cell(1.8), 'A3': cell(1.3, coverage=0.1)}, {'A2|A3': seam(0.6, 0.2, band=1.7, doubled=1)}, baseline, rules)
    assert not bad['passed']
    assert {k for k, v in bad['checks'].items() if not v} == {
        'A2_mae', 'A3_coverage', 'A2|A3_ownership', 'A2|A3_west_band', 'A2|A3_east_band', 'A2|A3_step_ground'}

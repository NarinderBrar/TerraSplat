"""Shared metric ENU ownership for independently anchored scenes."""
import numpy as np


def in_frame(points, source_to_ecef, target_to_ecef):
    transform = np.linalg.inv(np.asarray(target_to_ecef)) @ np.asarray(source_to_ecef)
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def cell_mask(points, cell):
    e0, e1 = cell['east_m']
    n0, n1 = cell['north_m']
    return ((points[:, 0] >= e0) & (points[:, 0] < e1)
            & (points[:, 1] >= n0) & (points[:, 1] < n1))


def owned_mask(points, georeference, grid, cell):
    shared = in_frame(points, georeference['enu_to_ecef'], grid['enu_to_ecef'])
    return cell_mask(shared, grid['cells'][cell])


def cell_corners(grid, cell, target_to_ecef):
    e0, e1 = grid['cells'][cell]['east_m']
    n0, n1 = grid['cells'][cell]['north_m']
    corners = np.array([[e0,n0,0], [e1,n0,0], [e1,n1,0], [e0,n1,0], [e0,n0,0]])
    return in_frame(corners, grid['enu_to_ecef'], target_to_ecef)


def partition_check(grid):
    """An analytic check, including the line itself; no sampling tolerance hides gaps."""
    west_name = 'centre' if 'centre' in grid['cells'] else 'west'
    west, east = (grid['cells'][k] for k in (west_name, 'east'))
    w0, w1 = west['east_m']
    e0, e1 = east['east_m']
    if not (w0 < w1 and e0 < e1 and west['north_m'][0] < west['north_m'][1]
            and east['north_m'][0] < east['north_m'][1]):
        raise ValueError('Invalid ownership rectangles')
    delta = e0 - w1
    same_north = west['north_m'] == east['north_m']
    return {'gap_width_m': max(delta, 0), 'overlap_width_m': max(-delta, 0),
            'north_ranges_match': same_north, 'boundary_east_m': w1,
            'passed': delta == 0 and same_north,
            'boundary_owner': 'east' if delta == 0 else None}

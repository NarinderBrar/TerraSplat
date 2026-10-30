"""Compute the pre-registered 3DEP shape check (configs/3dep-shape-check.json). Run with .venv-imagery."""

import json
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]


def load(directory):
    directory = ROOT / directory
    # DFC2019 .txt metadata: easting and northing of the grid's lower-left corner, size in pixels, resolution.
    west, south, size, resolution = np.loadtxt(directory / 'JAX_068_DSM.txt')
    dsm = rasterio.open(directory / 'JAX_068_DSM.tif').read(1).astype(float)
    cls = rasterio.open(directory / 'JAX_068_CLS.tif').read(1)
    return dsm, cls, west, south + dsm.shape[0] * resolution, resolution


def stats(difference):
    return {'pixels': int(difference.size), 'mae_m': float(np.mean(np.abs(difference))),
            'median_m': float(np.median(difference)), 'rmse_m': float(np.sqrt(np.mean(difference ** 2)))}


def main():
    rules = json.loads((ROOT / 'configs/3dep-shape-check.json').read_text())
    truth, truth_cls, west, north, resolution = load('artifacts/reference/dfc2019-jax068')
    key, key_cls, key_west, key_north, key_resolution = load('artifacts/reference/3dep-jax068-east/stable')
    assert resolution == key_resolution == 0.5
    rows, cols = np.indices(truth.shape)
    key_rows = rows + int(round((key_north - north) / 0.5))
    key_cols = cols + int(round((west - key_west) / 0.5))
    predicted, labels = key[key_rows, key_cols], key_cls[key_rows, key_cols]
    # Same pixels as the absolute qualification (pipeline/lidar_reference.py).
    labelled = ((labels != 65) & (labels != 9) & (truth_cls != 65) & (truth_cls != 9)
                & np.isfinite(predicted) & np.isfinite(truth))
    difference = predicted - truth
    offset = float(np.median(difference[labelled]))
    residual = difference - offset
    half_r, half_c = truth.shape[0] // 2, truth.shape[1] // 2
    quadrants = {}
    for name, (r, c) in {'nw': (slice(0, half_r), slice(0, half_c)), 'ne': (slice(0, half_r), slice(half_c, None)),
                         'sw': (slice(half_r, None), slice(0, half_c)), 'se': (slice(half_r, None), slice(half_c, None))}.items():
        quadrants[name] = float(np.median(difference[r, c][labelled[r, c]]))
    ground = labelled & (truth_cls == 2)
    result = {
        'rules_file': 'configs/3dep-shape-check.json', 'offset_m': offset,
        'all_labelled': stats(residual[labelled]), 'ground': stats(residual[ground]),
        'quadrant_offsets_m': quadrants,
        'quadrant_offset_spread_m': max(abs(v - offset) for v in quadrants.values()),
    }
    r = rules['rules']
    checks = {'min_pixels': result['all_labelled']['pixels'] >= r['min_pixels'],
              'all_labelled_mae': result['all_labelled']['mae_m'] <= r['all_labelled_mae_max_m'],
              'ground_mae': result['ground']['mae_m'] <= r['ground_mae_max_m'],
              'quadrant_offset_spread': result['quadrant_offset_spread_m'] <= r['quadrant_offset_spread_max_m']}
    result.update(checks=checks, passed=all(checks.values()))
    output = ROOT / 'artifacts/reference/3dep-jax068-east/shape-check.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

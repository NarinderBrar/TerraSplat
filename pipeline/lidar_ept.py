"""Fetch the lidar points inside a lon/lat box from a USGS 3DEP Entwine (EPT) dataset on AWS. Run with .venv-imagery.

Only the octree chunks overlapping the box are downloaded. Output keeps EPT coordinates (EPSG:3857 metres); EPT does
not state the vertical datum, so heights must be checked against an independent reference before use.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import urllib.request

import numpy as np

ROOT_URL = 'https://s3-us-west-2.amazonaws.com/usgs-lidar-public'


def get(url, attempts=4):
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return response.read()
        except OSError:
            if attempt == attempts - 1:
                raise


def node_bounds(cube, key):
    depth, x, y, z = map(int, key.split('-'))
    size = (np.asarray(cube[3:]) - np.asarray(cube[:3])) / 2 ** depth
    low = np.asarray(cube[:3]) + size * [x, y, z]
    return low, low + size


def chunk_keys(base, cube, box):
    """Keys of every non-empty node whose XY extent overlaps box = (xmin, ymin, xmax, ymax)."""
    keys, pending = [], ['0-0-0-0']
    while pending:
        hierarchy = json.loads(get(f'{base}/ept-hierarchy/{pending.pop()}.json'))
        for key, count in hierarchy.items():
            low, high = node_bounds(cube, key)
            if high[0] < box[0] or low[0] > box[2] or high[1] < box[1] or low[1] > box[3]:
                continue
            # -1 marks a subtree stored in its own file, which repeats this key with its real count.
            if count == -1:
                pending.append(key)
            elif count > 0:
                keys.append(key)
    return sorted(set(keys))


def read_chunk(base, key, box):
    import laspy

    data = get(f'{base}/ept-data/{key}.laz')
    points = laspy.read(io.BytesIO(data))
    x, y = np.asarray(points.x), np.asarray(points.y)
    inside = (x >= box[0]) & (x <= box[2]) & (y >= box[1]) & (y <= box[3])
    fields = {'x': x, 'y': y, 'z': np.asarray(points.z), 'classification': np.asarray(points.classification),
              'return_number': np.asarray(points.return_number), 'number_of_returns': np.asarray(points.number_of_returns),
              'intensity': np.asarray(points.intensity)}
    return key, hashlib.sha256(data).hexdigest(), {k: v[inside] for k, v in fields.items()}


def fetch(resource, lon_lat_box, output, workers=8):
    from pyproj import Transformer

    output = Path(output)
    if output.exists():
        raise ValueError('Use a new output directory')
    base = f'{ROOT_URL}/{resource}'
    ept = json.loads(get(f'{base}/ept.json'))
    if ept.get('srs', {}).get('horizontal') != '3857' or ept.get('dataType') != 'laszip':
        raise ValueError('Expected an EPSG:3857 laszip EPT dataset')
    to_mercator = Transformer.from_crs(4326, 3857, always_xy=True)
    west, south, east, north = lon_lat_box
    (x0, x1), (y0, y1) = to_mercator.transform([west, east], [south, north])
    box = (x0, y0, x1, y1)
    keys = chunk_keys(base, ept['bounds'], box)
    with ThreadPoolExecutor(workers) as pool:
        chunks = list(pool.map(lambda k: read_chunk(base, k, box), keys))
    fields = {k: np.concatenate([c[2][k] for c in chunks]) for k in chunks[0][2]}
    output.mkdir(parents=True)
    np.savez_compressed(output / 'points.npz', **fields)
    manifest = {
        'resource': resource, 'ept_url': f'{base}/ept.json', 'ept_points': ept['points'],
        'lon_lat_box': list(lon_lat_box), 'mercator_box': list(box), 'horizontal_crs': 'EPSG:3857',
        'z_units': 'metres', 'vertical_datum': 'not stated by EPT; verify against an independent reference',
        'points': int(len(fields['x'])),
        'classes': {int(c): int(n) for c, n in zip(*np.unique(fields['classification'], return_counts=True))},
        'z_range_m': [float(fields['z'].min()), float(fields['z'].max())],
        'chunks': {key: digest for key, digest, _ in chunks},
        'points_sha256': hashlib.sha256((output / 'points.npz').read_bytes()).hexdigest(),
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resource', required=True, help='EPT dataset name, e.g. FL_Peninsular_FDEM_Duval_2018')
    parser.add_argument('--box', required=True, type=float, nargs=4, metavar=('WEST', 'SOUTH', 'EAST', 'NORTH'))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    manifest = fetch(args.resource, args.box, args.output)
    print(json.dumps({k: manifest[k] for k in ('resource', 'points', 'classes', 'z_range_m')}, indent=2))


if __name__ == '__main__':
    main()

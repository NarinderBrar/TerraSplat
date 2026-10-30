"""Write the 1 km² JAX_068 grid (4 x 4 cells of 250 m) and one block config and approved selection per cell.

Rows A-D run north to south and columns 1-4 west to east, in the JAX_068 anchor ENU frame of configs/jax068-grid.json.
The grid spans east -500..500 m and north -750..250 m: one row south of centred, because the 2015-12-21 acquisition
(1040010016BADB00) does not cover north of +250 m. The anchor, and the DFC2019 lidar, sit where cells A2, A3, B2 and B3
meet. Every cell reuses the 19 training acquisitions and the May 2 holdout of configs/jax068-pilot-19v.json.
"""

import json
from pathlib import Path

import numpy as np
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
SIDE = 250
EAST = (-500, 500)
NORTH = (-750, 250)
# DFC2019 JAX_068 lidar (256 m about the anchor) reaches these cells only.
LIDAR_CELLS = {'A2', 'A3', 'B2', 'B3'}


def main():
    base = json.loads((ROOT / 'configs/jax068-grid.json').read_text())
    pilot = json.loads((ROOT / 'configs/jax068-pilot-19v.json').read_text())
    block = json.loads((ROOT / 'configs/jax068-block.json').read_text())
    enu_to_ecef = np.array(base['enu_to_ecef'])
    to_geodetic = Transformer.from_crs(4978, 4979, always_xy=True)
    cells = {}
    for row, north in enumerate(range(NORTH[1] - SIDE, NORTH[0] - 1, -SIDE)):
        for col, east in enumerate(range(EAST[0], EAST[1], SIDE)):
            name = f'{"ABCD"[row]}{col + 1}'
            lon, lat, _ = to_geodetic.transform(*(enu_to_ecef @ [east + SIDE / 2, north + SIDE / 2, 0, 1])[:3])
            block_path = f'configs/grid-1km/{name}-block.json'
            (ROOT / block_path).parent.mkdir(parents=True, exist_ok=True)
            (ROOT / block_path).write_text(json.dumps(block | {
                'longitude': lon, 'latitude': lat,
                'selection_note': f'1 km grid cell {name}: centre ({east + SIDE / 2:g}, {north + SIDE / 2:g}) m in '
                                  'jax068-grid anchor ENU. Same crop and height bracket as the pilot.',
            }, indent=2) + '\n')
            (ROOT / f'configs/grid-1km/{name}-19v.json').write_text(json.dumps(pilot | {'block': block_path}, indent=2) + '\n')
            cells[name] = {'east_m': [east, east + SIDE], 'north_m': [north, north + SIDE], 'block': block_path,
                           'centre_longitude': lon, 'centre_latitude': lat}
    grid = {key: base[key] for key in ('version', 'anchor', 'enu_to_ecef')} | {
        'id': 'jax068-1km', 'cell_side_m': SIDE, 'cells': cells, 'ownership': base['ownership'],
        'note': 'East -500..500 m, north -750..250 m: shifted one row south so all 20 acquisitions cover every cell '
                'plus its 20 m halo (1040010016BADB00 ends north of +250 m; smallest margin 70 px).',
    }
    (ROOT / 'configs/jax068-grid-1km.json').write_text(json.dumps(grid, indent=2) + '\n')
    write_jobs(cells)
    print(' '.join(cells))


def write_jobs(cells):
    """One pipeline.job file per cell, from the west split job; lidar truth only where DFC2019 reaches."""
    template = json.loads((ROOT / 'configs/jobs/west-19v-job.json').read_text())
    for name, cell in cells.items():
        key = name.lower()
        job = template | {
            'output_root': f'artifacts/jobs/grid-1km-{key}',
            'block_config': cell['block'],
            'approved_selection': f'configs/grid-1km/{name}-19v.json',
            'training_profile': 'configs/train-2015-pilot-19v-recipe.json',
            'ownership_grid': 'configs/jax068-grid-1km.json',
            'cell': name,
        }
        job['release'] = template['release'] | {
            'dataset_id': f'jax068-2015-grid-{key}', 'dataset_title': f'JAX_068 1 km grid {name} (historical)',
            'release_id': f'jax068-2015-grid-{key}-r1', 'title': f'JAX_068 1 km grid {name} · 19-view recipe',
        }
        if name in LIDAR_CELLS:
            job['lidar_truth'] = 'artifacts/reference/dfc2019-jax068'
        else:
            job.pop('lidar_truth', None)
        (ROOT / f'configs/jobs/grid-1km-{key}.json').write_text(json.dumps(job, indent=2) + '\n')


if __name__ == '__main__':
    main()

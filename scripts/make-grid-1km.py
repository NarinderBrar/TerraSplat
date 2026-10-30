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
    print(' '.join(cells))


if __name__ == '__main__':
    main()

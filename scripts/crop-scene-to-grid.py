"""Crop a prepared Skyfall scene's training views to the DFC2019 truth grid, adjusting each camera's principal point.

Keeps full resolution where the grid is small in a large image (the authors' 2048 px JAX crops), so LangSplat's
SAM + CLIP step sees the same ground detail as TerraSplat's own crops. Run with .venv-imagery:

    PYTHONPATH=. python scripts/crop-scene-to-grid.py --export DIR --points depth-points.npz --truth DIR --scene DIR \
        [--registration dsm-validation.json] --output NEW_DIR
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.dsm_validation import read_truth
from pipeline.semantics import top_surface
from pipeline.semantics_clip import aligned_georeference, cell_model_points, frame_projection

MARGIN_PX = 48


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ('--export', '--points', '--truth', '--scene', '--output'):
        parser.add_argument(name, required=True)
    parser.add_argument('--registration')
    args = parser.parse_args()
    scene, output = Path(args.scene), Path(args.output)
    if output.exists():
        raise ValueError('Use a new output directory')
    manifest = json.loads((Path(args.export) / 'export-manifest.json').read_text())
    geo, shift = aligned_georeference(manifest['georeference'], args.registration)
    truth = read_truth(args.truth, 'JAX_068')
    filled = top_surface(args.points, geo, truth, shift)[2]
    points = cell_model_points(truth, filled, geo)
    transforms = json.loads((scene / 'transforms_train.json').read_text())
    (output / 'images').mkdir(parents=True)
    (output / 'masks').mkdir()
    frames = []
    for frame in transforms['frames']:
        name = Path(frame['file_path']).stem
        P, (width, height) = frame_projection(frame, transforms)
        projected = np.column_stack((points, np.ones(len(points)))) @ P.T
        uv = projected[:, :2] / projected[:, 2:]
        x0, y0 = np.maximum(np.floor(uv.min(axis=0)).astype(int) - MARGIN_PX, 0)
        x1, y1 = np.minimum(np.ceil(uv.max(axis=0)).astype(int) + MARGIN_PX, (width, height))
        Image.open(scene / frame['file_path']).crop((x0, y0, x1, y1)).save(output / 'images' / f'{name}.png')
        np.save(output / 'masks' / f'{name}.npy', np.load(scene / 'masks' / f'{name}.npy')[y0:y1, x0:x1])
        frames.append(frame | {'file_path': f'./images/{name}.png', 'cx': frame['cx'] - x0, 'cy': frame['cy'] - y0,
                               'w': int(x1 - x0), 'h': int(y1 - y0), 'crop_origin_px': [int(x0), int(y0)]})
    (output / 'transforms_train.json').write_text(json.dumps(
        {k: v for k, v in transforms.items() if k not in ('frames', 'w', 'h')} | {'frames': frames}, indent=2) + '\n')
    print(f'{len(frames)} views cropped to the truth grid; sizes {sorted({(f["w"], f["h"]) for f in frames})}')


if __name__ == '__main__':
    main()

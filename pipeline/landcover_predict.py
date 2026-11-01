"""Predict OpenEarthMap probabilities from a trained local model.

 env -u PYTHONPATH .venv-landcover/bin/python -m pipeline.landcover_predict \
     --model artifacts/landcover/v5-oem-segformer-b2 --images /path/to/images --output /path/to/probs
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from pipeline.phase2 import digest

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
CLASSES = ['bareland', 'rangeland', 'developed space', 'road', 'tree', 'water', 'agriculture land', 'building']


def normalise(image):
    return torch.from_numpy(((image.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy())


def read_image(path):
    with rasterio.open(path) as source:
        return source.read([1, 2, 3]).transpose(1, 2, 0)


def window_starts(length, window, stride):
    end = max(0, length - window)
    return sorted(set(range(0, end + 1, stride)) | {end})


@torch.inference_mode()
def predict_probabilities(model, image_uint8_hwc, window=512, overlap=128, batch=4, flip_tta=False):
    if window < 1 or not 0 <= overlap < window or batch < 1:
        raise ValueError('Require window > overlap >= 0 and batch >= 1')
    image = image_uint8_hwc
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) == 0:
        raise ValueError('Expected a nonempty uint8 HWC RGB image')
    height, width = image.shape[:2]
    image = np.pad(image, ((0, max(0, window - height)), (0, max(0, window - width)), (0, 0)), mode='edge')
    positions = [(y, x) for y in window_starts(height, window, window - overlap)
                 for x in window_starts(width, window, window - overlap)]
    ramp = np.minimum(np.arange(window) + 1, np.arange(window, 0, -1)).astype(np.float32)
    weight = np.outer(ramp, ramp)
    weight /= weight.max()
    total = np.zeros(image.shape[:2] + (8,), np.float32)
    coverage = np.zeros(image.shape[:2], np.float32)
    device = next(model.parameters()).device
    training = model.training
    model.eval()
    try:
        for start in range(0, len(positions), batch):
            tiles = positions[start:start + batch]
            inputs = torch.stack([normalise(image[y:y + window, x:x + window]) for y, x in tiles]).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                logits = F.interpolate(model(pixel_values=inputs).logits.float(), (window, window), mode='bilinear',
                                       align_corners=False)
                probs = logits.softmax(1)
                if flip_tta:
                    flipped = model(pixel_values=inputs.flip(-1)).logits.float()
                    flipped = F.interpolate(flipped, (window, window), mode='bilinear', align_corners=False)
                    probs = (probs + flipped.flip(-1).softmax(1)) / 2
            probs = probs.permute(0, 2, 3, 1).cpu().numpy()
            for (y, x), tile in zip(tiles, probs):
                total[y:y + window, x:x + window] += tile * weight[..., None]
                coverage[y:y + window, x:x + window] += weight
    finally:
        model.train(training)
    return (total[:height, :width] / coverage[:height, :width, None]).astype(np.float16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--weights', default='best.safetensors')
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--window', type=int, default=512)
    parser.add_argument('--overlap', type=int, default=128)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--flip-tta', action='store_true')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    model = SegformerForSemanticSegmentation(SegformerConfig.from_pretrained(args.model, local_files_only=True))
    weights = args.model / args.weights
    model.load_state_dict(load_file(weights))
    model.to(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    files = []
    for path in sorted(args.images.iterdir()):
        if path.suffix.lower() not in ('.tif', '.tiff', '.png', '.jpg', '.jpeg'):
            continue
        target = args.output / f'{path.stem}_probs.npy'
        if target.exists():
            raise ValueError(f'Duplicate image stem: {path.stem}')
        probabilities = predict_probabilities(model, read_image(path), args.window, args.overlap, args.batch, args.flip_tta)
        np.save(target, probabilities)
        files.append({'image': str(path), 'image_sha256': digest(path), 'output': target.name,
                      'sha256': digest(target)})
    report = {'weights_sha256': digest(weights), 'model_config_sha256': digest(args.model / 'config.json'),
              'classes': CLASSES, 'window': args.window, 'overlap': args.overlap, 'batch': args.batch,
              'flip_tta': args.flip_tta, 'files': files}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

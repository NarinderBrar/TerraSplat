"""Train SegFormer on local OpenEarthMap images, without network access.

 env -u PYTHONPATH .venv-landcover/bin/python -m pipeline.landcover_train \
     --config configs/landcover-v5-train.json [--max-steps 2] [--resume]
"""

import argparse
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import rasterio
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from safetensors.torch import load_file, save_file
import transformers
from transformers import SegformerForSemanticSegmentation

from pipeline.landcover_predict import CLASSES, normalise, predict_probabilities, read_image
from pipeline.phase2 import digest


def read_split(root, split):
    root = Path(root)
    names = [line.strip() for line in (root / split).read_text().splitlines() if line.strip()]
    pairs, missing_images, missing_labels = [], 0, 0
    for name in names:
        region = Path(name).stem.rsplit('_', 1)[0]
        image, label = root / region / 'images' / name, root / region / 'labels' / name
        if not image.is_file():
            missing_images += 1
        elif not label.is_file():
            missing_labels += 1
        else:
            pairs.append((image, label))
    return pairs, {'listed': len(names), 'used': len(pairs), 'skipped': missing_images + missing_labels,
                   'missing_images': missing_images, 'missing_labels': missing_labels}


def map_labels(labels):
    if np.any(labels > 8):
        raise ValueError('OpenEarthMap labels must be codes 0..8')
    return np.where(labels == 0, 255, labels.astype(np.int64) - 1).astype(np.uint8)


def read_pair(pair):
    image = read_image(pair[0])
    with rasterio.open(pair[1]) as source:
        labels = map_labels(source.read(1))
    if image.shape[:2] != labels.shape:
        raise ValueError(f'Image/label shape mismatch: {pair[0]}')
    return image, labels


def augment(image, labels, crop, augmentations):
    height, width = labels.shape
    image = np.pad(image, ((0, max(0, crop - height)), (0, max(0, crop - width)), (0, 0)), mode='edge')
    labels = np.pad(labels, ((0, max(0, crop - height)), (0, max(0, crop - width))), constant_values=255)
    y, x = random.randrange(labels.shape[0] - crop + 1), random.randrange(labels.shape[1] - crop + 1)
    image, labels = image[y:y + crop, x:x + crop], labels[y:y + crop, x:x + crop]
    for name, axis in (('horizontal_flip', 1), ('vertical_flip', 0)):
        if name in augmentations and random.random() < 0.5:
            image, labels = np.flip(image, axis), np.flip(labels, axis)
    if 'rotate90' in augmentations:
        turns = random.randrange(4)
        image, labels = np.rot90(image, turns), np.rot90(labels, turns)
    return image.copy(), labels.copy()


class LandcoverDataset(Dataset):
    def __init__(self, pairs, crop, augmentations):
        self.pairs, self.crop, self.augmentations = pairs, crop, augmentations

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image, labels = augment(*read_pair(self.pairs[index]), self.crop, self.augmentations)
        return normalise(image), torch.from_numpy(labels.astype(np.int64))


def segmentation_loss(logits, labels):
    logits = F.interpolate(logits.float(), labels.shape[-2:], mode='bilinear', align_corners=False)
    losses = F.cross_entropy(logits, labels, ignore_index=255, reduction='sum')
    return losses / (labels != 255).sum().clamp_min(1)


def metrics_from_confusion(confusion):
    intersection = confusion.diagonal()
    union = confusion.sum(0) + confusion.sum(1) - intersection
    ious = [float(hit / count) if count else None for hit, count in zip(intersection, union)]
    present = [value for value in ious if value is not None]
    return {'per_class_iou': dict(zip(CLASSES, ious)),
            'miou': float(np.mean(present)) if present else None,
            'pixel_accuracy': float(intersection.sum() / confusion.sum()) if confusion.sum() else None}


def validate(model, pairs, settings):
    confusion = np.zeros((8, 8), np.int64)
    for pair in pairs:
        image, labels = read_pair(pair)
        predicted = predict_probabilities(model, image, **settings).argmax(-1)
        valid = labels != 255
        confusion += np.bincount(labels[valid].astype(np.int64) * 8 + predicted[valid], minlength=64).reshape(8, 8)
    return metrics_from_confusion(confusion)


def learning_rate(config, step):
    warmup, steps = config['warmup_steps'], config['steps']
    if step < warmup:
        factor = config['warmup_ratio'] + (1 - config['warmup_ratio']) * step / max(1, warmup)
    else:
        factor = max(0, (steps - step) / max(1, steps - warmup)) ** config['poly_power']
    return config['lr'] * factor


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Seeded, not bit-exact on CUDA: cross-entropy has no deterministic CUDA kernel, and deterministic upsampling needs
    # more memory than the 6 GB GPU has.
    torch.backends.cudnn.benchmark = False


def rng_state():
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': (state[0], state[1].tolist(), *state[2:]),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    numpy_state = state['numpy']
    np.random.set_state((numpy_state[0], np.array(numpy_state[1], np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_checkpoint(output, model, optimizer, scaler, report):
    weights = output / 'last.safetensors'
    temporary = output / 'last.tmp.safetensors'
    save_file(model.state_dict(), temporary)
    temporary.replace(weights)
    state = {'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(), 'step': report['step'],
             'rng': rng_state(), 'report': report, 'weights_sha256': digest(weights)}
    temporary = output / 'last.tmp.pt'
    torch.save(state, temporary)
    temporary.replace(output / 'last.pt')
    write_json(output / 'report.json', report)


def pretrained_digests(directory):
    paths = sorted(path for path in Path(directory).iterdir()
                   if path.suffix in ('.bin', '.safetensors'))
    if not paths:
        raise ValueError('Pretrained directory has no local weights')
    return {path.name: digest(path) for path in paths}


def check_config(config):
    for key in ('crop', 'batch_size', 'accumulation', 'steps', 'eval_interval', 'checkpoint_interval'):
        if config[key] < 1:
            raise ValueError(f'{key} must be positive')
    if not 0 <= config['warmup_steps'] < config['steps']:
        raise ValueError('warmup_steps must be in [0, steps)')
    unknown = set(config['augmentations']) - {'horizontal_flip', 'vertical_flip', 'rotate90'}
    if unknown:
        raise ValueError(f'Unknown augmentations: {unknown}')


def run(config_path, max_steps=None, resume=False):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    check_config(config)
    limit = config['steps'] if max_steps is None else min(max_steps, config['steps'])
    if limit < 1:
        raise ValueError('max_steps must be positive')
    output = Path(config['output_dir'])
    if output.exists() and not resume:
        raise ValueError('Use a new output directory or --resume')
    state = None
    if resume:
        if (output / 'report.json').exists() and json.loads((output / 'report.json').read_text())['finished']:
            raise ValueError('Refusing to overwrite a finished output directory')
        state = torch.load(output / 'last.pt', map_location='cpu', weights_only=True)
        if state['report']['config_sha256'] != digest(config_path):
            raise ValueError('Resume config differs from checkpoint')
        if state['weights_sha256'] != digest(output / 'last.safetensors'):
            raise ValueError('Checkpoint weights/state mismatch')
        if limit <= state['step']:
            raise ValueError('max_steps must exceed the saved step')
    train_pairs, train_counts = read_split(config['data_root'], config['train_split'])
    val_pairs, val_counts = read_split(config['data_root'], config['val_split'])
    if not train_pairs or not val_pairs:
        raise ValueError('Training and validation splits must contain labelled images')
    seed_everything(config['seed'])
    device = torch.device(config['device'])
    weights_digest = pretrained_digests(config['pretrained_dir'])
    model = SegformerForSemanticSegmentation.from_pretrained(
        config['pretrained_dir'], num_labels=8, id2label=dict(enumerate(CLASSES)),
        label2id={name: i for i, name in enumerate(CLASSES)}, ignore_mismatched_sizes=True,
        local_files_only=True, weights_only=True, semantic_loss_ignore_index=255).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'],
                                 betas=tuple(config['adam_betas']), eps=config['adam_eps'])
    scaler = torch.amp.GradScaler('cuda', enabled=config['amp'] and device.type == 'cuda')
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    split_digests = {name: digest(Path(config['data_root']) / config[f'{name}_split']) for name in ('train', 'val')}
    report = {'schema_version': 1, 'config_sha256': digest(config_path), 'config': config,
              'images': {'train': train_counts, 'val': val_counts}, 'split_sha256': split_digests,
              'pretrained_weight_sha256': weights_digest, 'git_commit': git.stdout.strip() or None,
              'torch': str(torch.__version__), 'transformers': transformers.__version__,
              'step': 0, 'best_step': None, 'best_val_metrics': None, 'val_metrics': None, 'finished': False}
    if state:
        if (state['report']['pretrained_weight_sha256'] != weights_digest or
                state['report']['split_sha256'] != split_digests or state['report']['images'] != report['images']):
            raise ValueError('Pretrained weights or data splits changed since checkpoint')
        model.load_state_dict(load_file(output / 'last.safetensors', device=str(device)))
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        report = state['report']
        restore_rng(state['rng'])
        log = output / 'train-log.jsonl'
        lines = [line for line in log.read_text().splitlines() if json.loads(line)['step'] <= report['step']]
        log.write_text(''.join(line + '\n' for line in lines))
    else:
        output.mkdir(parents=True)
        model.config.save_pretrained(output)
    dataset = LandcoverDataset(train_pairs, config['crop'], config['augmentations'])
    model.train()
    for step in range(report['step'] + 1, limit + 1):
        started = time.perf_counter()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        lr = learning_rate(config, step - 1)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(config['accumulation']):
            samples = [dataset[random.randrange(len(dataset))] for _ in range(config['batch_size'])]
            images, labels = [torch.stack(items).to(device) for items in zip(*samples)]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                loss = segmentation_loss(model(pixel_values=images).logits, labels)
            scaler.scale(loss / config['accumulation']).backward()
            loss_sum += loss.detach().item() / config['accumulation']
        scaler.step(optimizer)
        scaler.update()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        entry = {'step': step, 'loss': loss_sum, 'lr': lr, 'sec/step': time.perf_counter() - started,
                 'max_gpu_mib': torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0}
        with (output / 'train-log.jsonl').open('a') as stream:
            stream.write(json.dumps(entry, allow_nan=False) + '\n')
        report['step'] = step
        if step % config['eval_interval'] == 0 or step == limit:
            metrics = validate(model, val_pairs, config['validation'])
            report['val_metrics'] = metrics
            best = report['best_val_metrics']
            if metrics['miou'] is not None and (best is None or metrics['miou'] > best['miou']):
                save_file(model.state_dict(), output / 'best.safetensors')
                report.update(best_step=step, best_val_metrics=metrics)
        if step % config['checkpoint_interval'] == 0 or step == limit:
            report['finished'] = step == config['steps']
            save_checkpoint(output, model, optimizer, scaler, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.max_steps, args.resume), indent=2))


if __name__ == '__main__':
    main()

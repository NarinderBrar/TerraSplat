import json
import os
import random
import subprocess
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from safetensors.torch import load_file
import torch
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from pipeline.landcover_predict import predict_probabilities
from pipeline.landcover_train import (
    augment, learning_rate, map_labels, metrics_from_confusion, read_split, run, segmentation_loss,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_model():
    return SegformerForSemanticSegmentation(SegformerConfig(
        num_labels=8, depths=[1, 1, 1, 1], hidden_sizes=[8, 16, 24, 32],
        num_attention_heads=[1, 2, 3, 4], sr_ratios=[4, 2, 1, 1],
        decoder_hidden_size=16, drop_path_rate=0.1,
    ))


def write_tif(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, 'w', driver='GTiff', height=data.shape[1], width=data.shape[2],
                       count=data.shape[0], dtype=data.dtype, transform=from_origin(10, 20, 1, 1)) as sink:
        sink.write(data)


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / 'data'
    rows, cols = np.indices((35, 39))
    labels = ((rows + cols) % 9).astype(np.uint8)
    image = np.stack([labels * 28, rows.astype(np.uint8), cols.astype(np.uint8)])
    for number in (1, 2):
        write_tif(root / 'some_region/images' / f'some_region_{number}.tif', image)
        write_tif(root / 'some_region/labels' / f'some_region_{number}.tif', labels[None])
    (root / 'train.txt').write_bytes(b'some_region_1.tif\r\nsome_region_99.tif\r\n\r\n')
    (root / 'val.txt').write_bytes(b'some_region_2.tif\r\n')
    return root


def test_split_crlf_missing_images(data_root):
    pairs, counts = read_split(data_root, 'train.txt')
    assert pairs[0][0] == data_root / 'some_region/images/some_region_1.tif'
    assert counts == {'listed': 2, 'used': 1, 'skipped': 1, 'missing_images': 1, 'missing_labels': 0}


def test_label_mapping():
    np.testing.assert_array_equal(map_labels(np.arange(9, dtype=np.uint8)), [255, 0, 1, 2, 3, 4, 5, 6, 7])
    with pytest.raises(ValueError):
        map_labels(np.array([9], np.uint8))


@pytest.mark.parametrize('shape,crop', [((9, 11), 6), ((3, 5), 8)])
def test_augmentation_alignment(shape, crop):
    labels = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    image = np.repeat(labels[..., None], 3, axis=2)
    for seed in range(24):
        random.seed(seed)
        transformed, target = augment(image, labels, crop, ['horizontal_flip', 'vertical_flip', 'rotate90'])
        assert transformed.shape == (crop, crop, 3)
        valid = target != 255
        np.testing.assert_array_equal(transformed[..., 0][valid], target[valid])
        if crop > max(shape):
            assert valid.sum() == np.prod(shape)


@pytest.mark.parametrize('shape,tta', [((19, 23), False), ((67, 79), False), ((39, 51), True)])
def test_predict_probabilities(shape, tta):
    model = tiny_model().train()
    image = np.random.default_rng(42).integers(0, 256, (*shape, 3), dtype=np.uint8)
    probs = predict_probabilities(model, image, window=32, overlap=8, batch=2, flip_tta=tta)
    assert model.training
    assert probs.shape == (*shape, 8)
    assert probs.dtype == np.float16
    assert np.isfinite(probs).all()
    assert (probs >= 0).all()
    np.testing.assert_allclose(probs.astype(np.float32).sum(-1), 1, atol=0.001)


def test_metrics_ignore_and_schedule():
    confusion = np.zeros((8, 8), np.int64)
    confusion[0, 0], confusion[0, 1], confusion[1, 1] = 2, 1, 1
    metrics = metrics_from_confusion(confusion)
    assert metrics['pixel_accuracy'] == 0.75
    assert metrics['miou'] == pytest.approx((2 / 3 + 1 / 2) / 2)
    assert metrics['per_class_iou']['tree'] is None
    logits = torch.randn(1, 8, 2, 2, requires_grad=True)
    loss = segmentation_loss(logits, torch.full((1, 8, 8), 255, dtype=torch.long))
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    config = {'warmup_steps': 2, 'steps': 6, 'warmup_ratio': 0.1, 'poly_power': 1, 'lr': 1}
    assert [learning_rate(config, step) for step in (0, 2, 4, 6)] == [0.1, 1, 0.5, 0]


def test_train_resume_and_prediction_cli(tmp_path, data_root):
    pretrained = tmp_path / 'pretrained'
    tiny_model().save_pretrained(pretrained, safe_serialization=True)
    config = json.loads(open('configs/landcover-v5-train.json').read())
    output = tmp_path / 'training'
    config.update(data_root=str(data_root), pretrained_dir=str(pretrained), output_dir=str(output),
                  crop=32, batch_size=2, accumulation=2, steps=4, warmup_steps=1,
                  device='cpu', eval_interval=2, checkpoint_interval=2,
                  validation={'window': 32, 'overlap': 8, 'batch': 2, 'flip_tta': False})
    path = tmp_path / 'train.json'
    path.write_text(json.dumps(config))
    report = run(path, max_steps=2)
    assert report['step'] == 2
    assert not report['finished']
    assert report['images']['train']['missing_images'] == 1
    assert len(report['config_sha256']) == 64
    assert report['best_step'] in (2,)
    assert report['val_metrics']['miou'] is not None
    assert len(report['pretrained_weight_sha256']['model.safetensors']) == 64
    for name in ('best.safetensors', 'last.safetensors', 'last.pt', 'config.json', 'train-log.jsonl', 'report.json'):
        assert (output / name).is_file()
    state = torch.load(output / 'last.pt', weights_only=True)
    assert state['step'] == 2
    assert state['optimizer']['state']
    assert state['rng']['torch'].numel() > 0
    with pytest.raises(ValueError, match='new output'):
        run(path)
    config['lr'] *= 2
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='config differs'):
        run(path, resume=True)
    config['lr'] /= 2
    path.write_text(json.dumps(config))
    env = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    command = [sys.executable, '-m', 'pipeline.landcover_train', '--config', str(path), '--resume']
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    resumed = load_file(output / 'last.safetensors')
    report = json.loads((output / 'report.json').read_text())
    assert report['finished'] and report['step'] == 4
    assert [json.loads(line)['step'] for line in (output / 'train-log.jsonl').read_text().splitlines()] == [1, 2, 3, 4]
    with pytest.raises(ValueError, match='finished'):
        run(path, resume=True)
    config['output_dir'] = str(tmp_path / 'continuous')
    path.write_text(json.dumps(config))
    run(path)
    continuous = load_file(tmp_path / 'continuous/last.safetensors')
    for name in resumed:
        torch.testing.assert_close(resumed[name], continuous[name], rtol=0, atol=0)
    prediction_dir = tmp_path / 'predictions'
    command = [sys.executable, '-m', 'pipeline.landcover_predict', '--model', str(output), '--images',
               str(data_root / 'some_region/images'), '--output', str(prediction_dir),
               '--window', '32', '--overlap', '8', '--batch', '2', '--device', 'cpu']
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert len(list(prediction_dir.glob('*_probs.npy'))) == 2
    prediction_report = json.loads((prediction_dir / 'report.json').read_text())
    assert len(prediction_report['files']) == 2
    assert len(prediction_report['weights_sha256']) == 64

"""Kaggle script kernel: verify OpenEarthMap against our checksums, then run pipeline.landcover_train.

Inputs (attached datasets): the private bundle from scripts/kaggle-landcover-bundle.py, a public OpenEarthMap copy and,
to continue a stopped run, the output of an earlier version of this kernel (its last.pt is resumed).
Training stops itself before Kaggle's 12 h limit so the checkpoints are kept as the version's output.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

INPUT, WORK = Path('/kaggle/input'), Path('/kaggle/working')
TIME_LIMIT_S = 11 * 3600


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def find_one(pattern):
    found = sorted(INPUT.rglob(pattern))
    if len(found) != 1:
        raise SystemExit(f'Expected one {pattern} under {INPUT}, found {len(found)}')
    return found[0]


def kind(path):
    parts = [part.lower() for part in path.parts]
    labels, images = any('label' in part for part in parts), any('image' in part for part in parts)
    return 'labels' if labels and not images else 'images' if images and not labels else None


def prepare_data(bundle):
    manifest = json.loads((bundle / 'oem-sha256.json').read_text())
    names = set()
    for split in ('train.txt', 'val.txt'):
        names |= {line.strip() for line in (bundle / split).read_text().splitlines() if line.strip()}
    # Only names with an image can be trained on; xBD labels without images may be absent from a copy.
    names = {name for name in names if name in manifest['images']}
    found = {'images': {}, 'labels': {}}
    for path in INPUT.rglob('*.tif'):
        group = kind(path.relative_to(INPUT))
        if group and path.name in names:
            if path.name in found[group]:
                raise SystemExit(f'Duplicate {group} file {path.name} in the OpenEarthMap copy')
            found[group][path.name] = path
    files = [(group, name, path) for group, paths in found.items() for name, path in paths.items()]
    with ThreadPoolExecutor(8) as pool:
        digests = list(pool.map(lambda item: sha256(item[2]), files))
    root = WORK / 'oem'
    for (group, name, path), value in zip(files, digests):
        if manifest[group].get(name) != value:
            raise SystemExit(f'{group}/{name} differs from the checked Zenodo copy')
        target = root / name.rsplit('_', 1)[0] / group / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(path)
    for split in ('train.txt', 'val.txt'):
        shutil.copy2(bundle / split, root / split)
    counts = {group: len(paths) for group, paths in found.items()}
    print(f'OpenEarthMap: {len(files)} files match the checksums; found {counts} of {len(names)} images', flush=True)
    if counts != {'images': len(names), 'labels': len(names)}:
        raise SystemExit('The OpenEarthMap copy is missing images or labels that our download has')


def main():
    started = time.time()
    bundle = find_one('oem-sha256.json').parent
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.53.3', 'rasterio==1.4.*',
                    'safetensors'], check=True)
    prepare_data(bundle)
    shutil.copytree(bundle / 'mit-b2', WORK / 'mit-b2')
    config = bundle / 'code' / 'configs' / 'landcover-v5-kaggle.json'
    output = Path(json.loads(config.read_text())['output_dir'])
    previous = [path.parent for path in INPUT.rglob('last.pt') if path.parent.name == output.name]
    command = [sys.executable, '-m', 'pipeline.landcover_train', '--config', str(config)]
    if previous:
        if len(previous) > 1:
            raise SystemExit(f'More than one earlier run attached: {previous}')
        shutil.copytree(previous[0], output)
        command.append('--resume')
        print(f'Resuming from {previous[0]}', flush=True)
    subprocess.run(['nvidia-smi'], check=False)
    env = dict(os.environ, PYTHONPATH=str(bundle / 'code'))
    process = subprocess.Popen(command, cwd=bundle / 'code', env=env)
    try:
        process.wait(timeout=max(60, TIME_LIMIT_S - (time.time() - started)))
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait()
        print('Stopped before the Kaggle time limit; attach this output to the next version to resume.', flush=True)
        return
    if process.returncode:
        raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()

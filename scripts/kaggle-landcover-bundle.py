"""Build the private Kaggle dataset for v5 land-cover training (code, settings, pretrained weights, data checksums).

    env -u PYTHONPATH python3 scripts/kaggle-landcover-bundle.py <OpenEarthMap_wo_xBD dir> <mit-b2 dir> kaggle/landcover/bundle
    env -u PYTHONPATH kaggle datasets create -p kaggle/landcover/bundle -r zip      # first time (private by default)
    env -u PYTHONPATH kaggle kernels push -p kaggle/landcover --accelerator NvidiaTeslaT4

The notebook (kaggle/landcover/train.py) checks every OpenEarthMap file it trains on against oem-sha256.json, so a
public Kaggle copy of the data can be used instead of uploading ours.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

CODE = ['pipeline/landcover_train.py', 'pipeline/landcover_predict.py', 'pipeline/phase2.py',
        'pipeline/georeference.py', 'configs/landcover-v5-kaggle.json', 'kaggle/landcover/train.py']


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    data, pretrained, output = (Path(arg) for arg in sys.argv[1:4])
    repo = Path(__file__).resolve().parents[1]
    output.mkdir(parents=True)
    for name in CODE:
        target = output / 'code' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / name, target)
    for name in ('config.json', 'pytorch_model.bin', 'REVISION'):
        (output / 'mit-b2').mkdir(exist_ok=True)
        shutil.copy2(pretrained / name, output / 'mit-b2' / name)
    for name in ('train.txt', 'val.txt'):
        shutil.copy2(data / name, output / name)
    files = {'images': {}, 'labels': {}}
    for kind, entries in files.items():
        for path in sorted(data.glob(f'*/{kind}/*.tif')):
            if path.name in entries:
                raise ValueError(f'Duplicate file name {path.name}')
            entries[path.name] = sha256(path)
    manifest = {'source': 'Zenodo 7223446 OpenEarthMap.zip (md5 64155d1dc9d3b68536063f79878e1a67), OpenEarthMap_wo_xBD',
                **files}
    (output / 'oem-sha256.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
    metadata = {'title': 'terrasplat-landcover-bundle', 'id': 'npsbrar1986/terrasplat-landcover-bundle',
                'licenses': [{'name': 'other'}]}
    (output / 'dataset-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f"{len(files['images'])} images, {len(files['labels'])} labels -> {output}")


if __name__ == '__main__':
    main()

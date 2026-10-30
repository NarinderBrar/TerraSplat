"""Collect measured split quality for the publisher; no training or setting selection."""
import argparse
import hashlib
import json
from pathlib import Path


def collect(block):
    def read(path):
        return json.loads(Path(path).read_text())
    run = f'artifacts/stage1/{block}-2015-19v-recipe-01'
    tile_dir = f'artifacts/tiles/{block}-2015-19v-recipe-01-r1'
    lidar = f'artifacts/validation/{block}-2015-19v-recipe-01-lidar/dsm-validation.json'
    evaluation = f'{run}/evaluation/evaluation-report.json'
    seam_path = 'artifacts/validation/phase3-split-seam-check.json'
    dsm, ev, seam = read(lidar), read(evaluation), read(seam_path)
    if not seam['decision']['passed']:
        raise ValueError('Pre-registered split rules failed')
    selection = read(f'artifacts/imagery/{block}-19v/selection.json')
    holdout_ids = {v['catalog_id'] for v in selection['holdout']}
    records = [r for r in ev['records']['final'] if r['role'] == 'holdout']
    if len(records) != 1 or {r['catalog_id'] for r in records} != holdout_ids:
        raise ValueError('Unexpected holdout acquisitions')
    tiles, plan = read(f'{tile_dir}/tiles-report.json'), read(f'{tile_dir}/tile-plan.json')
    quality = {
        'registered_lidar_mae_all_m':dsm['registered']['all_labelled']['mae_m'],
        'registered_lidar_mae_ground_m':dsm['registered']['ground']['mae_m'],
        'registered_lidar_mae_buildings_m':dsm['registered']['building']['mae_m'],
        'registered_lidar_mae_vegetation_m':dsm['registered']['vegetation']['mae_m'],
        'registered_lidar_rmse_m':dsm['registered']['all_labelled']['rmse_m'],
        'holdout_psnr_db':records[0]['psnr_db'],
        'horizontal_offset_m':dsm['registration']['splats_offset_from_lidar_m']['horizontal'],
        'completeness':dsm['registered']['all_labelled']['completeness'],
        'splats':plan['crop']['owned_splats'], 'tiles':len(tiles['tiles']),
        'glb_bytes':tiles['totals']['glbBytes']}
    paths = [lidar,evaluation,seam_path,f'{tile_dir}/tile-plan.json',f'{tile_dir}/tiles-report.json']
    result = {'status':'measured','quality':quality,
              'inputs_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}}
    with Path(f'training/phase3-{block}-result.json').open('x') as output:
        output.write(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--block',choices=('west','east'),required=True)
    collect(parser.parse_args().block)

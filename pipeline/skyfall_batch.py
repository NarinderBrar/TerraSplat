"""Serial, restartable Stage 1 scene jobs. Source scripts/stage1-env.sh first."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from pipeline.phase2 import digest
from pipeline.reference_scene import prepare
from pipeline.skyfall_scenes import LABEL, place, verify_frame
from pipeline.publish import publish

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def command(args, log):
    with Path(log).open('w') as stream:
        subprocess.run([str(a) for a in args], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def evidence(scene, run, placed, tiles, depth):
    training = json.loads((run / 'training-report.json').read_text())
    evaluation = json.loads((run / 'evaluation/evaluation-report.json').read_text())
    manifest = json.loads((placed / 'export-manifest.json').read_text())
    plan = json.loads((tiles / 'tile-plan.json').read_text())
    encoded = json.loads((tiles / 'tiles-report.json').read_text())
    quality = dict.fromkeys(['registered_lidar_mae_all_m', 'registered_lidar_mae_ground_m',
                            'registered_lidar_mae_buildings_m', 'registered_lidar_mae_vegetation_m',
                            'registered_lidar_rmse_m', 'horizontal_offset_m', 'completeness'])
    holdout = [r['psnr_db'] for r in evaluation['records']['final'] if r['role'] == 'holdout']
    quality.update(holdout_psnr_db=float(np.mean(holdout)) if holdout else None, splats=plan['crop']['owned_splats'],
                   tiles=len(plan['tiles']), glb_bytes=encoded['totals']['glbBytes'])
    compact = {'scene': scene['scene'], 'label': LABEL, 'quality': quality,
               'dataset_revision': '82bcd8ae044afcf3182d335a47d0694ba2b4daa2',
               'download_manifest_sha256': digest(ROOT.parent / 'Ref/Skyfall-GS/data/download-manifest.json'),
               'training': {k: training[k] for k in ('final_splats', 'elapsed_seconds', 'peak_allocated_bytes',
                                                    'profile_sha256', 'train_catalog_ids', 'holdout_catalog_ids')},
               'evaluation_sha256': digest(run / 'evaluation/evaluation-report.json'),
               'holdout_used_for_training_or_selection': False,
               'placement': manifest['placement'], 'anchor': manifest['georeference']['anchor'],
               'crop': plan['crop'], 'levels': plan['levels'], 'encoding': encoded}
    path = Path('training') / (scene['dataset_id'] + '.json')
    write(path, compact)
    config = {'dataset_id': scene['dataset_id'], 'dataset_title': 'Skyfall ' + scene['scene'],
              'region': scene['region'], 'release_id': scene['dataset_id'] + '-stage1-r1',
              'title': f"Skyfall {scene['scene']} · TerraSplat Stage 1", 'description': LABEL,
              'tiles_dir': str(tiles), 'stage1_run_dir': str(run), 'evidence': [str(path)],
              'capture_dates': {'train': ['unavailable'], 'holdout': ['unavailable']},
              'quality_sources': {k: {'file': str(path), 'pointer': '/quality/' + k} for k in quality},
              'ground_points': str(depth), 'source_label': "TerraSplat Stage 1 · authors' prepared Skyfall data",
              'accuracy_label': 'TerraScope visual registration ~1.3–3.8 m; vertical approximate'}
    config_path = Path('configs/releases') / (scene['dataset_id'] + '.json')
    write(config_path, config)
    return config_path


def run_scene(scene, recipe, prepare_only=False):
    name = scene['scene']
    workspace = Path('artifacts/skyfall') / name
    workspace.mkdir(parents=True, exist_ok=True)
    source = Path(scene['source'])
    registration = Path(scene['registration'])
    verify_frame(source, json.loads(registration.read_text()))
    prepared = workspace / 'scene'
    if not prepared.exists():
        prepare(source, prepared, 'artifacts/reference/dfc2019-jax068/JAX_068_DSM.txt', 'stage1-pseudo-depth')
    if prepare_only:
        return {'scene': name, 'status': 'prepared'}
    free = shutil.disk_usage(workspace).free
    if free < 5 * 1024**3:
        raise OSError(f'Disk space below 5 GiB: {free} bytes free; GPU work stopped')
    stage1 = ROOT / '.venv-stage1/bin/python'
    run = Path('artifacts/stage1') / (scene['dataset_id'] + '-recipe-01')
    if not run.exists():
        command([stage1, '-m', 'pipeline.training', '--scene', prepared, '--profile', recipe, '--output', run],
                workspace / 'training-cli.log')
    trained = json.loads((run / 'training-report.json').read_text())
    if trained['status'] != 'trained':
        raise ValueError('Existing scene run is incomplete or failed; preserve for diagnosis')
    if not (run / 'evaluation/evaluation-report.json').exists():
        command([stage1, '-m', 'pipeline.evaluation', '--run', run], workspace / 'evaluation-cli.log')
    if not (run / 'export/export-manifest.json').exists():
        command([stage1, '-m', 'pipeline.export', '--run', run, '--gsd', '0.5'], workspace / 'export-cli.log')
    depth = workspace / 'depth'
    if not (depth / 'depth-report.json').exists():
        command([stage1, '-m', 'pipeline.render_depth', '--run', run, '--output', depth, '--half-extent', '280'],
                workspace / 'depth-cli.log')
    placed = workspace / 'placed'
    if not placed.exists():
        place(run, source, registration, placed, depth / 'depth-points.npz', scene.get('core_model_xy', (0, 0)))
    profile = json.loads(Path('configs/tiles-skyfall.json').read_text())
    profile['block_half_extent_m'] = scene['core_half_extent_m']
    profile_path = workspace / 'tiles-profile.json'
    write(profile_path, profile)
    tiles = Path('artifacts/tiles') / (scene['dataset_id'] + '-stage1-r1')
    if not (tiles / 'tile-plan.json').exists():
        command([ROOT / '.venv-imagery/bin/python', '-m', 'pipeline.tiling', '--export', placed,
                 '--output', tiles, '--profile', profile_path], workspace / 'tiling-cli.log')
    if not (tiles / 'tiles-report.json').exists():
        command(['node', 'scripts/encode-tiles.mjs', tiles], workspace / 'encoding-cli.log')
    config = evidence(scene, run, placed, tiles, placed / 'depth-points.npz')
    release_id = scene['dataset_id'] + '-stage1-r1'
    if not (Path('artifacts/releases') / release_id).exists():
        publish(config, Path('artifacts/releases'), True)
    return {'scene': name, 'status': 'published', 'release_id': release_id}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes', nargs='*')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    config = json.loads(Path('configs/skyfall-scenes.json').read_text())
    status_path = Path('artifacts/skyfall/batch-status.json')
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    for scene in config['scenes']:
        if scene['scene'] == 'JAX_068' or (args.scenes and scene['scene'] not in args.scenes):
            continue
        name = scene['scene']
        status[name] = {'scene': name, 'status': 'preparing' if args.prepare_only else 'running',
                        'started': time.time()}
        write(status_path, status)
        print(name, status[name]['status'], flush=True)
        try:
            status[name] = run_scene(scene, config['recipe'], args.prepare_only)
        except OSError as exc:
            status[name] = {'scene': name, 'status': 'failed', 'error': str(exc)}
            write(status_path, status)
            if 'Disk space below' in str(exc):
                print(str(exc), flush=True)
                break
        except Exception as exc:
            status[name] = {'scene': name, 'status': 'failed', 'error': str(exc)}
        write(status_path, status)
        print(json.dumps(status[name]), flush=True)


if __name__ == '__main__':
    main()

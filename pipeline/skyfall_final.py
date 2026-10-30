"""Place, tile and publish the Skyfall-GS authors' finished models. Run with .venv-imagery.

TerraScope measured its visual registrations on these same models, so they apply without retraining.
Downloads stay read-only; each scene writes a derived export manifest under artifacts/skyfall/<scene>/.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from pipeline.export import TILER_RULES, apply_transform, column_major, read_attributes, round_trip, validate_attributes
from pipeline.phase2 import digest
from pipeline.publish import publish
from pipeline.skyfall_scenes import placed_georeference, verify_frame
from pipeline.tiling import plan

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "jayinnn/Skyfall-GS-ply"
REVISION = "cdd48ec6ad5177afba2e57d28ddb86118143009c"
MODELS = ROOT.parent / "Ref/Skyfall-GS/data/final-ply-cdd48ec"
LABEL = ("Skyfall-GS authors' finished model (jayinnn/Skyfall-GS-ply, Apache-2.0); "
         "visually registered by TerraScope (about 1.3-3.8 m); vertical approximate; "
         "not produced from TerraSplat's raw-imagery pipeline")


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def export(scene, ply, output):
    """Write a placed export manifest for a finished PLY, plus ENU centres of its opaque splats for ground."""
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new placed export directory")
    source, registration_path = ROOT / scene['source'], ROOT / scene['registration']
    registration = json.loads(registration_path.read_text())
    frame = verify_frame(source, registration)
    geo = placed_georeference(registration, scene.get('core_model_xy', (0, 0)))
    attributes = read_attributes(ply)
    count = len(attributes['x'])
    validation = validate_attributes(attributes, count)
    trip = round_trip(attributes, geo, 0.5)
    if not trip['passed']:
        raise ValueError("Placed transform round trip failed")
    output.mkdir(parents=True)
    centres = np.stack([attributes[a] for a in 'xyz'], 1)
    opaque = 1 / (1 + np.exp(-attributes['opacity'])) > 0.5
    np.savez_compressed(output / 'surface-points.npz', enu=apply_transform(geo['model_to_enu'], centres[opaque]))
    manifest = {
        'schema_version': 1, 'status': 'exported', 'assumed_anchor': False,
        'run_directory': None, 'camera_scene': str(source.resolve()),
        'source': {'repository': REPOSITORY, 'revision': REVISION, 'file': Path(ply).name},
        'fused_ply': {'path': str(Path(ply).resolve()), 'sha256': digest(ply), 'bytes': Path(ply).stat().st_size},
        'splat_count': count,
        'sh_degree': int(round(np.sqrt(sum(n.startswith('f_rest_') for n in attributes) / 3 + 1))) - 1,
        'georeference': geo,
        'tileset': {'root_transform': column_major(geo['model_to_ecef']),
                    'root_transform_layout': 'column-major 4x4, as 3D Tiles expects',
                    'root_transform_equation': 'root.transform = model_to_ecef', 'rules': TILER_RULES},
        'validation': validation, 'round_trip': trip,
        'placement': {'registration_sha256': digest(registration_path), 'frame_check': frame,
                      'label': LABEL, 'vertical_method': geo['vertical_method']},
    }
    write(output / 'export-manifest.json', manifest)
    return manifest


def run_scene(scene):
    name = scene['scene']
    workspace = ROOT / 'artifacts/skyfall' / name
    placed = workspace / 'final-placed'
    if not (placed / 'export-manifest.json').exists():
        export(scene, MODELS / f'{name}_final.ply', placed)
    release_id = scene['dataset_id'] + '-final-r1'
    tiles = ROOT / 'artifacts/tiles' / release_id
    if not (tiles / 'tile-plan.json').exists():
        profile = json.loads((ROOT / 'configs/tiles-skyfall.json').read_text())
        profile['block_half_extent_m'] = scene['core_half_extent_m']
        write(workspace / 'final-tiles-profile.json', profile)
        plan(placed, tiles, profile)
    if not (tiles / 'tiles-report.json').exists():
        with (workspace / 'final-encoding-cli.log').open('w') as log:
            subprocess.run(['node', 'scripts/encode-tiles.mjs', tiles], cwd=ROOT, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
    manifest = json.loads((placed / 'export-manifest.json').read_text())
    tile_plan = json.loads((tiles / 'tile-plan.json').read_text())
    encoded = json.loads((tiles / 'tiles-report.json').read_text())
    quality = dict.fromkeys(['registered_lidar_mae_all_m', 'registered_lidar_mae_ground_m',
                             'registered_lidar_mae_buildings_m', 'registered_lidar_mae_vegetation_m',
                             'registered_lidar_rmse_m', 'horizontal_offset_m', 'completeness', 'holdout_psnr_db'])
    quality.update(splats=tile_plan['crop']['owned_splats'], tiles=len(tile_plan['tiles']),
                   glb_bytes=encoded['totals']['glbBytes'])
    evidence = Path('training') / f"{scene['dataset_id']}-final.json"
    write(ROOT / evidence, {'scene': name, 'label': LABEL, 'quality': quality,
                            'model': manifest['source'] | {'sha256': manifest['fused_ply']['sha256'],
                                                          'splats': manifest['splat_count']},
                            'holdout_used_for_training_or_selection': 'unknown; trained by the authors',
                            'placement': manifest['placement'], 'anchor': manifest['georeference']['anchor'],
                            'crop': tile_plan['crop'], 'levels': tile_plan['levels'], 'encoding': encoded})
    config = {'dataset_id': scene['dataset_id'], 'dataset_title': 'Skyfall ' + name, 'region': scene['region'],
              'release_id': release_id, 'title': f'Skyfall {name} · authors\' finished model', 'description': LABEL,
              'tiles_dir': str(tiles.relative_to(ROOT)), 'stage1_run_dir': os.path.relpath(MODELS / f'{name}_final.ply', ROOT),
              'evidence': [str(evidence)],
              'capture_dates': {'train': ['unavailable'], 'holdout': ['unavailable']},
              'quality_sources': {k: {'file': str(evidence), 'pointer': '/quality/' + k} for k in quality},
              'ground_points': str((placed / 'surface-points.npz').relative_to(ROOT)),
              'ground_surface': 'opaque splat-centre surface',
              'source_label': "Skyfall-GS authors' finished model",
              'accuracy_label': 'TerraScope visual registration ~1.3–3.8 m; vertical approximate'}
    config_path = ROOT / 'configs/releases' / f'{release_id}.json'
    write(config_path, config)
    if not (ROOT / 'artifacts/releases' / release_id).exists():
        publish(config_path, ROOT / 'artifacts/releases', True)
    return {'scene': name, 'status': 'published', 'release_id': release_id, 'splats': quality['splats'],
            'tiles': quality['tiles'], 'glb_bytes': quality['glb_bytes']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes', nargs='*')
    args = parser.parse_args()
    scenes = json.loads((ROOT / 'configs/skyfall-scenes.json').read_text())['scenes']
    for scene in scenes:
        if not args.scenes or scene['scene'] in args.scenes:
            print(json.dumps(run_scene(scene)), flush=True)


if __name__ == '__main__':
    main()

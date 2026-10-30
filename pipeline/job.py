"""Offline one-block jobs with immutable attempts and content-checked reuse."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.' + path.name,
                                     suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def inventory(path):
    path = Path(path)
    if not path.exists():
        return None
    if path.is_file():
        return {'.': file_hash(path)}
    return {str(p.relative_to(path)): file_hash(p) for p in sorted(path.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}


@dataclass
class Stage:
    name: str
    commands: list[list[str]]
    environment: str = 'sfm'
    gpu: bool = False
    files: list[str] = field(default_factory=list)
    entries: dict = field(default_factory=dict)
    copies: list[tuple[str, str]] = field(default_factory=list)
    # Commands use {work} for this attempt and {stage} for committed upstream outputs.


def stages_for(job):
    job = {**job, 'source_cache': job.get('source_cache', 'artifacts/imagery/raw')}
    stages = []
    def add(name, args, env='imagery', gpu=False, keys=(), copies=()):
        entries = {k: job[k] for k in keys if k in job}
        files = [str(v) for k, v in entries.items() if k not in ('cell', 'release') and isinstance(v, str)]
        stages.append(Stage(name, [args], env, gpu, files, entries, list(copies)))
    def module(env, name, *args):
        return [f'.venv-{env}/bin/python', '-m', 'pipeline.' + name, *map(str, args)]
    add('select', module('imagery', 'acquisitions', 'scan', job['catalogue'], '--block', job['block_config'], '--offline', '--output', '{work}/acquisitions.json'), keys=('catalogue', 'block_config', 'approved_selection'))
    stages[-1].commands.append(module('imagery', 'acquisitions', 'select', '{work}/acquisitions.json', '--approved', job['approved_selection'], '--output', '{work}/selection.json'))
    add('sources', module('imagery', 'sources', '{select}/selection.json', '--offline', '--max-total-gib', '30', '--destination', job.get('source_cache', 'artifacts/imagery/raw'), '--output', '{work}/sources.json'), keys=('source_cache',))
    add('block_inputs', module('imagery', 'block_inputs', '{select}/selection.json', '{sources}/sources.json', '--output', '{work}/inputs'))
    add('satellite_sfm', module('imagery', 'satellite_sfm', '{select}/selection.json', '--output', '{work}/calibration'))
    add('reconstruct', module('sfm', 'reconstruct', '--inputs', '{block_inputs}/inputs', '--calibration', '{satellite_sfm}/calibration', '--approved', job['approved_selection'], '--output', '{work}/sfm'), 'sfm', keys=('approved_selection',))
    add('skyfall_inputs', module('sfm', 'skyfall_inputs', '--reconstruction', '{reconstruct}/sfm', '--inputs', '{block_inputs}/inputs', '--approved', job['approved_selection'], '--output', '{work}/scene'), 'sfm', keys=('approved_selection',))
    add('input_check', ['.venv-stage1/bin/python', 'scripts/check-skyfall-inputs.py', '--scene', '{work}/scene'], 'stage1', copies=(('{skyfall_inputs}/scene', 'scene'),))
    for name, gpu in [('cpu_preflight', False), ('gpu_preflight', True)]:
        args = module('stage1', 'phase2', '{skyfall_inputs}/scene/phase2-manifest.json', '--output', '{work}/preflight-report.json')
        if not gpu:
            args.append('--skip-environment')
        add(name, args, 'stage1', gpu)
    add('training', module('stage1', 'training', '--scene', '{skyfall_inputs}/scene', '--profile', job['training_profile'], '--output', '{work}/run'), 'stage1', True, ('training_profile',))
    add('evaluation', module('stage1', 'evaluation', '--run', '{work}/run'), 'stage1', True, copies=(('{training}/run', 'run'),))
    add('export', module('stage1', 'export', '--run', '{training}/run', '--selection', '{select}/selection.json', '--output', '{work}/export'), 'stage1', True)
    add('render_depth', module('stage1', 'render_depth', '--run', '{training}/run', '--export', '{export}/export', '--output', '{work}/depth'), 'stage1', True)
    ownership = ['--grid', job['ownership_grid'], '--cell', job['cell']] if job.get('ownership_grid') else []
    if bool(job.get('ownership_grid')) != bool(job.get('cell')):
        raise ValueError('ownership_grid and cell must be supplied together')
    if job.get('lidar_truth'):
        add('dsm_validation', module('imagery', 'dsm_validation', '--export', '{export}/export', '--points', '{render_depth}/depth/depth-points.npz', '--truth', job['lidar_truth'], *ownership, '--output', '{work}/validation'), keys=('lidar_truth', 'ownership_grid', 'cell'))
    add('tiling', module('imagery', 'tiling', '--export', '{export}/export', '--profile', job['tile_profile'], *ownership, '--output', '{work}/tiles'), keys=('tile_profile', 'ownership_grid', 'cell'))
    add('encoding', ['node', 'scripts/encode-tiles.mjs', '{work}/tiles'], 'node', copies=(('{tiling}/tiles', 'tiles'),))
    add('publish', module('sfm', 'publish', '--recover-identical', '--config', '{work}/release.json', '--releases', '{work}/releases'), 'sfm', keys=('release',))
    return stages


class Runner:
    def __init__(self, job, stages=None, *, dry_run=False, until=None, from_stage=None,
                 no_gpu=False, publish_to=None, activate=False):
        self.job = job
        if stages is None and job.get('version') != 1:
            raise ValueError('Job version must be 1')
        self.root = Path(job['output_root']).absolute()
        if self.root.is_symlink():
            raise ValueError('Output root cannot be a symlink')
        self.root = self.root.resolve()
        if self.root == ROOT or self.root in ROOT.parents:
            raise ValueError('Output root must be a dedicated job directory')
        self.stages = stages if stages is not None else stages_for(job)
        names = [s.name for s in self.stages]
        for name in (until, from_stage):
            if name and name not in names:
                raise ValueError(f'Unknown stage: {name}')
        self.dry_run, self.until, self.from_stage = dry_run, until, from_stage
        self.no_gpu, self.publish_to, self.activate = no_gpu, publish_to, activate
        self.path = self.root / 'job-state.json'
        self.state = read(self.path) if self.path.exists() else {'version': 1, 'stages': {}, 'partials': []}
        self.initialize_records()

    def initialize_records(self):
        for stage in self.stages:
            record = self.state['stages'].setdefault(stage.name, {})
            defaults = dict(status='pending', commands=stage.commands, environment=stage.environment,
                            gpu=stage.gpu, start_time=None, end_time=None, log_path=None,
                            input_hash=None, outputs=None, output_hash=None)
            for key, value in defaults.items():
                record.setdefault(key, value)
            if record['status'] == 'pending' and record['start_time'] is None:
                record.update(commands=stage.commands, environment=stage.environment, gpu=stage.gpu)

    def save(self):
        atomic(self.path, self.state)

    def safe(self, path):
        path = Path(path).absolute()
        if not path.resolve().is_relative_to(self.root) or path == self.root:
            raise ValueError(f'Path outside output root: {path}')
        for parent in (path, *path.parents):
            if parent == self.root:
                break
            if parent.is_symlink():
                raise ValueError(f'Symlink in output path: {path}')
        return path

    def cleanup(self, path):
        path = Path(path)
        # Only successfully created, recorded direct attempt containers qualify.
        if (str(path) not in self.state['partials'] or path.parent != self.root / 'attempts'
                or not path.name.endswith('.partial')):
            raise ValueError(f'Refusing to remove unowned partial: {path}')
        self.safe(path)
        if path.exists():
            shutil.rmtree(path)
        self.state['partials'].remove(str(path))
        self.save()

    def input_hash(self, stage, upstream):
        files = {p: inventory(ROOT / p) for p in stage.files}
        if stage.name == 'select':
            files['metadata_cache'] = inventory(ROOT / '.cache/core3d-metadata')
        code = []
        for cmd in stage.commands:
            if '-m' in cmd:
                code.append(ROOT / (cmd[cmd.index('-m') + 1].replace('.', '/') + '.py'))
            code.extend(ROOT / a for a in cmd if a.endswith(('.py', '.mjs')) and '{' not in a)
        if stage.environment == 'stage1':
            code.append(ROOT / 'scripts/stage1-env.sh')
        return digest({'entries': stage.entries, 'files': files, 'commands': stage.commands,
                       'preparation_code': file_hash(Path(__file__)) if stage.name == 'publish' else None,
                       'code': {str(p): inventory(p) for p in code},
                       'environment': stage.environment, 'upstream': upstream,
                       'publish': [self.publish_to, self.activate] if stage.name == 'publish' else None})

    def execute(self, cmd, stage, log):
        env = {**os.environ, 'PYTHONPATH': '', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1'}
        argv = cmd
        if stage.environment == 'stage1':
            argv = ['bash', '-c', 'source scripts/stage1-env.sh && exec "$@"', 'job', *cmd]
        process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                try:
                    code = process.wait(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    if shutil.disk_usage(self.root).free < 20 * 1024**3:
                        raise OSError('Less than 20 GiB free during stage; job stopped')
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
            raise
        if code:
            raise subprocess.CalledProcessError(code, argv)

    def prepare_release(self, work, paths):
        # Quality pointers are explicitly supplied by the job and refer to fresh stage reports.
        config = json.loads(json.dumps(self.job['release']))
        def expand(value):
            if isinstance(value, str):
                if '{dsm_validation}' in value and not self.job.get('lidar_truth'):
                    return value
                return value.format(**paths)
            if isinstance(value, list):
                return [expand(v) for v in value]
            if isinstance(value, dict):
                return {expand(k): expand(v) for k, v in value.items()}
            return value
        config = expand(config)
        plan = read(Path(paths['encoding']) / 'tiles/tile-plan.json')
        config.setdefault('quality_values', {})['tiles'] = len(plan['tiles'])
        selection = read(Path(paths['select']) / 'selection.json')
        config['capture_dates'] = {role: [v['capture_time'] for v in selection[role]]
                                   for role in ('train', 'holdout')}
        # Use the acquisition role rather than depending on evaluation record order.
        evaluation_path = paths['evaluation'] + '/run/evaluation/evaluation-report.json'
        evaluation = read(evaluation_path)
        holdouts = [(i, r) for i, r in enumerate(evaluation['records']['final']) if r['role'] == 'holdout']
        if len(holdouts) != 1 or holdouts[0][1]['catalog_id'] != selection['holdout'][0]['catalog_id']:
            raise ValueError('Expected exactly the approved holdout in evaluation')
        config['quality_sources']['holdout_psnr_db'] = {'file': evaluation_path,
                                                       'pointer': f'/records/final/{holdouts[0][0]}/psnr_db'}
        if evaluation_path not in config['evidence']:
            config['evidence'].append(evaluation_path)
        if not self.job.get('lidar_truth'):
            lidar_keys = {'registered_lidar_mae_all_m', 'registered_lidar_mae_ground_m',
                          'registered_lidar_mae_buildings_m', 'registered_lidar_mae_vegetation_m',
                          'registered_lidar_rmse_m', 'horizontal_offset_m', 'completeness'}
            for key in lidar_keys:
                config['quality_sources'].pop(key, None)
                config['quality_values'][key] = None
            config['evidence'] = [p for p in config['evidence'] if 'dsm_validation' not in p]
            config.update(lidar_truth_unavailable=True, accuracy_label='No independent lidar truth supplied')
        config.update(tiles_dir=paths['encoding'] + '/tiles', stage1_run_dir=paths['training'] + '/run',
                      ground_points=paths['render_depth'] + '/depth/depth-points.npz')
        atomic(work / 'release.json', config)

    def release_alias(self, output):
        if self.publish_to:
            return
        alias = self.root / 'releases'
        target = Path(output) / 'releases'
        if alias.is_symlink() and alias.resolve() == target:
            return
        if alias.exists() and not alias.is_symlink():
            raise ValueError(f'Refusing to replace existing release directory: {alias}')
        temporary = self.root / '.releases-link'
        if temporary.exists() or temporary.is_symlink():
            raise ValueError(f'Unexpected release link staging path: {temporary}')
        temporary.symlink_to(target, target_is_directory=True)
        temporary.replace(alias)

    def run(self):
        if self.dry_run:
            return self.run_stages()
        self.root.mkdir(parents=True, exist_ok=True)
        self.safe(self.path)
        with (self.root / '.job.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Re-read after acquiring the lock.
            if self.path.exists():
                self.state = read(self.path)
                self.initialize_records()
            self.save()
            # A kill can land between rename, alias creation and the done-state write.
            for record in self.state['stages'].values():
                if record.get('committing'):
                    final = self.safe(record['output'])
                    partial = final.with_name(final.name + '.partial')
                    if final.exists() and inventory(final) == record['outputs']:
                        if not partial.is_symlink():
                            if partial.exists():
                                raise ValueError(f'Commit collision: {partial}')
                            partial.symlink_to(final.name, target_is_directory=True)
                        elif partial.resolve() != final:
                            raise ValueError(f'Commit alias changed: {partial}')
                        if str(partial) in self.state['partials']:
                            self.state['partials'].remove(str(partial))
                        if record.get('end_time') is None:
                            ended = datetime.now(timezone.utc)
                            record.update(end_time=ended.isoformat(), elapsed_seconds=(
                                ended - datetime.fromisoformat(record['start_time'])).total_seconds())
                        record.update(status='done', committing=False)
                        self.save()
            return self.run_stages()

    def run_stages(self):
        paths, upstream = {}, {}
        dirty = False
        for index, stage in enumerate(self.stages):
            if self.no_gpu and stage.gpu:
                print(f'STOP before {stage.name} [GPU]', flush=True)
                break
            record = self.state['stages'][stage.name]
            hashed = self.input_hash(stage, upstream)
            dirty |= stage.name == self.from_stage
            reusable = (not dirty and record['status'] == 'done' and record.get('input_hash') == hashed
                        and Path(record['output'] + '.partial').is_symlink()
                        and Path(record['output'] + '.partial').resolve() == Path(record['output'])
                        and inventory(record['output']) == record.get('outputs'))
            if reusable and record.get('published_output'):
                reusable = inventory(record['published_output']) == record.get('published_outputs')
            dirty |= not reusable
            if reusable:
                paths[stage.name] = record['output']
                upstream[stage.name] = record['output_hash']
                print(f'REUSE {stage.name}', flush=True)
                commands = record['commands']
                if stage.name == 'publish' and not self.dry_run:
                    self.release_alias(record['output'])
            else:
                attempt = stage.name + '-' + (uuid.uuid4().hex[:12] if not self.dry_run else 'NEXT')
                final = self.root / 'attempts' / attempt
                work = final.with_name(final.name + '.partial')
                context = {**paths, 'work': str(work)}
                commands = [[a.format(**context) for a in c] for c in stage.commands]
                if stage.name == 'publish':
                    if self.publish_to:
                        commands[0][-1] = str(Path(self.publish_to).absolute())
                    if self.activate:
                        commands[0].append('--activate')
                print(f'RUN {stage.name}' + (' [GPU]' if stage.gpu else ''), flush=True)
                if not self.dry_run:
                    if shutil.disk_usage(self.root).free < 20 * 1024**3:
                        raise OSError('Less than 20 GiB free; job stopped')
                    for downstream in self.stages[index + 1:]:
                        self.state['stages'][downstream.name].update(status='pending', stale=True)
                    self.save()
                    for old in list(self.state['partials']):
                        if Path(old).name.startswith(stage.name + '-'):
                            self.cleanup(old)
                    self.safe(work)
                    work.parent.mkdir(parents=True, exist_ok=True)
                    record.update(status='running', commands=commands, environment=stage.environment, gpu=stage.gpu,
                                  input_hash=hashed, start_time=datetime.now(timezone.utc).isoformat(), end_time=None,
                                  log_path=str(self.root / (attempt + '.log')), output=str(final), stale=False,
                                  committing=False)
                    self.save()
                    started = time.monotonic()
                    try:
                        work.mkdir()
                        self.state['partials'].append(str(work))
                        self.save()
                        for source, target in stage.copies:
                            shutil.copytree(source.format(**context), work / target)
                        if stage.name == 'publish':
                            self.prepare_release(work, context)
                        with Path(record['log_path']).open('w') as log:
                            for cmd in commands:
                                self.execute(cmd, stage, log)
                        if stage.name == 'publish' and self.publish_to:
                            release_id = read(work / 'release.json')['release_id']
                            published = Path(self.publish_to).absolute() / release_id
                            record.update(published_output=str(published), published_outputs=inventory(published))
                        outputs = inventory(work)
                        # Record rename intent before commit, so an interrupted rename is recoverable.
                        record.update(outputs=outputs, output_hash=digest(outputs), committing=True)
                        self.save()
                        if final.exists() or final.is_symlink():
                            raise ValueError(f'Refusing to replace completed output: {final}')
                        work.rename(final)
                        # Reports contain absolute paths: retain a compatibility alias after commit.
                        work.symlink_to(final.name, target_is_directory=True)
                        self.state['partials'].remove(str(work))
                        record.update(status='done', committing=False)
                        if stage.name == 'publish':
                            self.release_alias(final)
                    except BaseException as exc:
                        record.update(status='failed', error=str(exc))
                        with Path(record['log_path']).open('a') as log:
                            log.write(f'\nRunner: {type(exc).__name__}: {exc}\n')
                        raise
                    finally:
                        record.update(end_time=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started)
                        self.save()
                    print(f"DONE {stage.name} {record['elapsed_seconds']:.2f}s log={record['log_path']}", flush=True)
                paths[stage.name] = str(final)
                upstream[stage.name] = record.get('output_hash', 'pending')
            if self.dry_run:
                for cmd in commands:
                    prefix = 'source scripts/stage1-env.sh && ' if stage.environment == 'stage1' else ''
                    print('  ' + prefix + shlex.join(cmd))
            if stage.name == self.until:
                break
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--until')
    parser.add_argument('--from', dest='from_stage')
    parser.add_argument('--no-gpu', action='store_true')
    parser.add_argument('--publish-to')
    parser.add_argument('--activate', action='store_true')
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        return Runner(read(args.job), **{k: v for k, v in vars(args).items() if k != 'job'}).run()
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f'Job stopped: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())

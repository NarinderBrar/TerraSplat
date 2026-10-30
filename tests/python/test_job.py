import json
from pathlib import Path
import shlex
import signal
import subprocess
import sys

import pytest

from pipeline.job import ROOT, Runner, Stage, read, stages_for


def fake(name, *, gpu=False, files=(), script=None):
    return Stage(name, [[sys.executable, '-c', script or
                        "from pathlib import Path; import sys; Path(sys.argv[1], 'result').write_text('ok')",
                        '{work}']], gpu=gpu, files=list(map(str, files)))


def fake_job(tmp_path):
    job = {'output_root': str(tmp_path / 'job')}
    return job, [fake('a'), fake('b'), fake('c', gpu=True)]


def test_full_run_reuse_and_output_change(tmp_path, capsys):
    job, stages = fake_job(tmp_path)
    Runner(job, stages).run()
    state_path = Path(job['output_root']) / 'job-state.json'
    before = read(state_path)
    capsys.readouterr()
    Runner(job, stages).run()
    assert capsys.readouterr().out.splitlines() == ['REUSE a', 'REUSE b', 'REUSE c']
    assert read(state_path) == before
    output = Path(before['stages']['b']['output'])
    (output / 'result').write_text('tampered')
    Runner(job, stages).run()
    after = read(state_path)
    assert after['stages']['a'] == before['stages']['a']
    assert after['stages']['b']['output'] != str(output)
    assert (output / 'result').read_text() == 'tampered'
    assert after['stages']['c']['output'] != before['stages']['c']['output']


def test_input_change_invalidates_downstream(tmp_path):
    job, stages = fake_job(tmp_path)
    config = tmp_path / 'config.json'
    config.write_text('one')
    stages[1].files = [str(config)]
    Runner(job, stages).run()
    before = read(Path(job['output_root']) / 'job-state.json')['stages']
    config.write_text('two')
    Runner(job, stages).run()
    after = read(Path(job['output_root']) / 'job-state.json')['stages']
    assert after['a'] == before['a']
    assert all(after[n]['output'] != before[n]['output'] for n in ('b', 'c'))


def test_until_from_no_gpu_and_dry_run(tmp_path, capsys):
    job, stages = fake_job(tmp_path)
    Runner(job, stages, dry_run=True).run()
    assert not Path(job['output_root']).exists()
    Runner(job, stages, until='a').run()
    state = read(Path(job['output_root']) / 'job-state.json')
    assert state['stages']['a']['status'] == 'done'
    assert state['stages']['b']['status'] == 'pending'
    Runner(job, stages, no_gpu=True).run()
    state = read(Path(job['output_root']) / 'job-state.json')
    assert state['stages']['b']['status'] == 'done'
    assert state['stages']['c']['status'] == 'pending'
    first = state['stages']['a']['output']
    second = state['stages']['b']['output']
    Runner(job, stages, from_stage='b', until='b').run()
    state = read(Path(job['output_root']) / 'job-state.json')
    assert state['stages']['a']['output'] == first
    assert state['stages']['b']['output'] != second
    with pytest.raises(ValueError, match='Unknown stage'):
        Runner(job, stages, until='missing')


def test_failure_stops_and_leaves_log(tmp_path):
    job, stages = fake_job(tmp_path)
    stages[1] = fake('b', script="import sys; print('failure detail', flush=True); sys.exit(7)")
    with pytest.raises(subprocess.CalledProcessError):
        Runner(job, stages).run()
    state = read(Path(job['output_root']) / 'job-state.json')
    assert [state['stages'][n]['status'] for n in 'abc'] == ['done', 'failed', 'pending']
    assert 'failure detail' in Path(state['stages']['b']['log_path']).read_text()


def test_killed_running_stage_cleans_only_owned_partial_and_resumes(tmp_path):
    root = tmp_path / 'job'
    marker = tmp_path / 'once'
    script = ("from pathlib import Path; import os, signal, sys; "
              "p=Path(sys.argv[1]); (p/'result').write_text('ok'); "
              f"m=Path({str(marker)!r}); "
              "first=not m.exists(); m.touch(); "
              "os.kill(os.getppid(), signal.SIGKILL) if first else None")
    code = "from pipeline.job import Runner, Stage; Runner(" + repr({'output_root': str(root)}) + ", [Stage('a', " + repr([[sys.executable, '-c', script, '{work}']]) + ")]).run()"
    process = subprocess.run([sys.executable, '-c', code], cwd=ROOT)
    assert process.returncode == -signal.SIGKILL
    state = read(root / 'job-state.json')
    assert state['stages']['a']['status'] == 'running'
    partial = Path(state['partials'][0])
    assert partial.is_dir() and not partial.is_symlink()
    sentinel = root / 'unowned.partial'
    sentinel.mkdir()
    (sentinel / 'keep').write_text('keep')
    Runner({'output_root': str(root)}, [fake('a', script=script)]).run()
    assert not partial.exists()
    assert (sentinel / 'keep').read_text() == 'keep'
    assert read(root / 'job-state.json')['stages']['a']['status'] == 'done'


@pytest.mark.parametrize('kind', ['outside', 'unowned', 'symlink', 'completed'])
def test_refuses_unsafe_cleanup(tmp_path, kind):
    job, stages = fake_job(tmp_path)
    runner = Runner(job, stages)
    runner.root.mkdir()
    (runner.root / 'attempts').mkdir()
    outside = tmp_path / 'valuable.partial'
    outside.mkdir()
    (outside / 'keep').write_text('valuable')
    target = runner.root / 'attempts' / 'a-test.partial'
    if kind == 'outside':
        target = outside
        runner.state['partials'].append(str(target))
    elif kind == 'symlink':
        target.symlink_to(outside, target_is_directory=True)
        runner.state['partials'].append(str(target))
    elif kind == 'completed':
        target = target.with_suffix('')
        target.mkdir()
        runner.state['partials'].append(str(target))
    else:
        target.mkdir()
    with pytest.raises(ValueError):
        runner.cleanup(target)
    assert (outside / 'keep').read_text() == 'valuable'
    assert target.exists()


def test_commit_rename_interruption_is_recovered(tmp_path):
    job, stages = fake_job(tmp_path)
    Runner(job, stages, until='a').run()
    root = Path(job['output_root'])
    state = read(root / 'job-state.json')
    record = state['stages']['a']
    final = Path(record['output'])
    alias = final.with_name(final.name + '.partial')
    alias.unlink()  # Emulate kill after directory rename, before alias and state commit.
    record.update(status='running', committing=True)
    state['partials'].append(str(alias))
    (root / 'job-state.json').write_text(json.dumps(state))
    Runner(job, stages, until='a').run()
    assert alias.resolve() == final
    assert read(root / 'job-state.json')['stages']['a']['output'] == str(final)


def test_recipe_equivalence():
    job = read(ROOT / 'configs/jobs/central-19v.json')
    stages = {s.name: s for s in stages_for(job)}
    recipe = read(ROOT / 'training/result-2015-19v.json')['commands']
    checks = [('select', 0, 1), ('sources', 1, 0), ('block_inputs', 2, 0),
              ('satellite_sfm', 3, 0), ('reconstruct', 4, 0), ('skyfall_inputs', 5, 0),
              ('training', 10, 0), ('evaluation', 11, 0), ('export', 12, 0)]
    def normalize(args):
        result = []
        skip = False
        for token in args:
            if skip:
                skip = False
                continue
            # Explicit offline cache, export output, and recipe profile are the documented additions.
            if token in ('--destination', '--output'):
                skip = True
                continue
            if token == '--offline':
                continue
            if token.startswith(('artifacts/', '{')):
                token = '<path>'
            result.append(token)
        return result
    for name, index, command_index in checks:
        assert normalize(stages[name].commands[command_index]) == normalize(shlex.split(recipe[index]))
    parts = recipe[6].split(' && ')
    for name, expected in [('input_check', parts[1]), ('gpu_preflight', parts[2])]:
        assert normalize(stages[name].commands[0][1:]) == normalize(shlex.split(expected)[1:])
        assert stages[name].environment == 'stage1'
    assert '--skip-environment' in stages['cpu_preflight'].commands[0]
    assert '--offline' in stages['sources'].commands[0]
    assert job['approved_selection'] == 'configs/jax068-pilot-19v.json'
    assert job['training_profile'] == 'configs/train-2015-pilot-19v-recipe.json'
    assert all(stages[n].gpu for n in ('gpu_preflight', 'training', 'evaluation', 'export', 'render_depth'))


def test_from_marks_previously_completed_downstream_pending(tmp_path):
    job, stages = fake_job(tmp_path)
    Runner(job, stages).run()
    Runner(job, stages, from_stage='b', until='b').run()
    assert read(Path(job['output_root']) / 'job-state.json')['stages']['c']['status'] == 'pending'


def test_copy_failure_has_log_and_preserves_upstream(tmp_path):
    job, stages = fake_job(tmp_path)
    stages[1].copies = [('{a}/missing', 'copy')]
    with pytest.raises(FileNotFoundError):
        Runner(job, stages).run()
    records = read(Path(job['output_root']) / 'job-state.json')['stages']
    assert records['a']['status'] == 'done'
    assert records['b']['status'] == 'failed'
    assert 'FileNotFoundError' in Path(records['b']['log_path']).read_text()


def test_release_paths_activation_and_optional_truth(tmp_path):
    job = read(ROOT / 'configs/jobs/central-19v.json')
    job['output_root'] = str(tmp_path / 'job')
    runner = Runner(job, dry_run=True, publish_to=str(tmp_path / 'review'), activate=True)
    # Preparation resolves measured holdout by role, even when it is the first record.
    paths = {}
    for name in ('encoding', 'evaluation', 'training', 'select', 'render_depth'):
        paths[name] = str(tmp_path / name)
    encoding = Path(paths['encoding']) / 'tiles'
    encoding.mkdir(parents=True)
    (encoding / 'tile-plan.json').write_text(json.dumps({'tiles': [{'id': 'r'}]}))
    evaluation = Path(paths['evaluation']) / 'run/evaluation'
    evaluation.mkdir(parents=True)
    (evaluation / 'evaluation-report.json').write_text(json.dumps({'records': {'final': [
        {'role': 'holdout', 'catalog_id': 'holdout', 'psnr_db': 17.2},
        {'role': 'train', 'catalog_id': 'train', 'psnr_db': 25}]}}))
    selection = Path(paths['select'])
    selection.mkdir()
    (selection / 'selection.json').write_text(json.dumps({
        'train': [{'capture_time': '2015-05-01', 'catalog_id': 'train'}],
        'holdout': [{'capture_time': '2015-05-02', 'catalog_id': 'holdout'}]}))
    work = tmp_path / 'publish'
    work.mkdir()
    job.pop('lidar_truth')
    assert 'dsm_validation' not in [s.name for s in stages_for(job)]
    runner.prepare_release(work, paths)
    config = read(work / 'release.json')
    assert config['quality_sources']['holdout_psnr_db']['pointer'] == '/records/final/0/psnr_db'
    assert config['quality_values']['tiles'] == 1
    assert config['quality_values']['registered_lidar_mae_all_m'] is None
    assert config['accuracy_label'] == 'No independent lidar truth supplied'
    assert not any('dsm_validation' in p for p in config['evidence'])
    assert config['capture_dates']['holdout'] == ['2015-05-02']
    assert config['tiles_dir'] == str(encoding)


def test_committed_alias_cannot_be_cleaned_even_if_recorded(tmp_path):
    job, stages = fake_job(tmp_path)
    Runner(job, stages, until='a').run()
    runner = Runner(job, stages)
    final = Path(runner.state['stages']['a']['output'])
    alias = final.with_name(final.name + '.partial')
    runner.state['partials'].append(str(alias))
    with pytest.raises(ValueError, match='Symlink'):
        runner.cleanup(alias)
    assert (final / 'result').read_text() == 'ok'


def test_missing_committed_alias_invalidates_stage(tmp_path):
    job, stages = fake_job(tmp_path)
    Runner(job, stages).run()
    records = read(Path(job['output_root']) / 'job-state.json')['stages']
    old = records['b']['output']
    Path(old + '.partial').unlink()
    Runner(job, stages).run()
    after = read(Path(job['output_root']) / 'job-state.json')['stages']
    assert after['a'] == records['a']
    assert after['b']['output'] != old
    assert (Path(old) / 'result').read_text() == 'ok'


def test_publication_flags_and_default_isolation(tmp_path, capsys):
    job = {'output_root': str(tmp_path / 'job')}
    stage = Stage('publish', [[sys.executable, '-c', 'pass', '--releases', '{work}/releases']])
    Runner(job, [stage], dry_run=True).run()
    default_plan = capsys.readouterr().out
    assert str(tmp_path / 'job') in default_plan
    assert '--activate' not in default_plan
    target = tmp_path / 'review'
    Runner(job, [stage], dry_run=True, publish_to=str(target), activate=True).run()
    plan = capsys.readouterr().out
    assert f'--releases {target} --activate' in plan
    assert not target.exists()


def test_disk_guard_stops_before_stage(tmp_path, monkeypatch):
    from collections import namedtuple
    job, stages = fake_job(tmp_path)
    usage = namedtuple('Usage', 'free')
    monkeypatch.setattr('pipeline.job.shutil.disk_usage', lambda path: usage(19 * 1024**3))
    with pytest.raises(OSError, match='20 GiB'):
        Runner(job, stages).run()
    state = read(Path(job['output_root']) / 'job-state.json')
    assert state['stages']['a']['status'] == 'pending'
    assert not (Path(job['output_root']) / 'attempts').exists()


def test_collision_never_claims_or_deletes_foreign_partial(tmp_path, monkeypatch):
    from types import SimpleNamespace
    job, stages = fake_job(tmp_path)
    runner = Runner(job, stages, until='a')
    foreign = runner.root / 'attempts/a-collision000.partial'
    foreign.mkdir(parents=True)
    (foreign / 'keep').write_text('foreign')
    monkeypatch.setattr('pipeline.job.uuid.uuid4', lambda: SimpleNamespace(hex='collision000anything'))
    with pytest.raises(FileExistsError):
        runner.run()
    assert read(runner.path)['partials'] == []
    with pytest.raises(FileExistsError):
        Runner(job, stages, until='a').run()
    assert (foreign / 'keep').read_text() == 'foreign'


def test_collision_never_replaces_completed_empty_directory(tmp_path, monkeypatch):
    from types import SimpleNamespace
    job, stages = fake_job(tmp_path)
    final = Path(job['output_root']) / 'attempts/a-collision000'
    final.mkdir(parents=True)
    inode = final.stat().st_ino
    monkeypatch.setattr('pipeline.job.uuid.uuid4', lambda: SimpleNamespace(hex='collision000anything'))
    with pytest.raises(ValueError, match='Refusing to replace completed'):
        Runner(job, stages, until='a').run()
    assert final.stat().st_ino == inode
    assert list(final.iterdir()) == []


def test_default_source_cache_is_an_input_without_mutating_job():
    job = read(ROOT / 'configs/jobs/central-19v.json')
    job.pop('source_cache')
    source = next(s for s in stages_for(job) if s.name == 'sources')
    assert source.entries['source_cache'] == 'artifacts/imagery/raw'
    assert 'artifacts/imagery/raw' in source.files
    assert 'source_cache' not in job


def test_commit_recovery_records_completion_time(tmp_path):
    job, stages = fake_job(tmp_path)
    Runner(job, stages, until='a').run()
    root = Path(job['output_root'])
    state = read(root / 'job-state.json')
    record = state['stages']['a']
    record.update(status='running', committing=True, end_time=None)
    record.pop('elapsed_seconds')
    (root / 'job-state.json').write_text(json.dumps(state))
    Runner(job, stages, until='a').run()
    recovered = read(root / 'job-state.json')['stages']['a']
    assert recovered['status'] == 'done'
    assert recovered['end_time'] is not None
    assert recovered['elapsed_seconds'] >= 0

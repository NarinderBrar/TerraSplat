"""Live progress of pipeline.job runs: one line per job, with the running stage and how long it has run.

    python3 scripts/job-progress.py 'artifacts/jobs/grid-1km-*'          # once
    watch -n 5 python3 scripts/job-progress.py 'artifacts/jobs/grid-1km-*'  # refreshing
"""

from datetime import datetime, timezone
import glob
import json
from pathlib import Path
import sys


def minutes(seconds):
    return f'{int(seconds // 60)}m{int(seconds % 60):02d}s'


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else 'artifacts/jobs/*'
    now = datetime.now(timezone.utc)
    for root in sorted(glob.glob(pattern)):
        path = Path(root) / 'job-state.json'
        if not path.is_file():
            continue
        stages = json.loads(path.read_text())['stages']
        done = sum(s['status'] == 'done' for s in stages.values())
        line = f'{Path(root).name:24} {done:2}/{len(stages)} done'
        for name, stage in stages.items():
            if stage['status'] in ('running', 'failed'):
                started = datetime.fromisoformat(stage['start_time'])
                line += f'  {stage["status"].upper()} {name} {minutes((now - started).total_seconds())}'
                # COLMAP and other tools write their own logs inside the attempt; show the last line.
                logs = sorted(Path(stage['output'] + '.partial').glob('**/logs/*.log'), key=lambda p: p.stat().st_mtime)
                if logs and stage['status'] == 'running':
                    tail = [l.strip() for l in logs[-1].read_text(errors='replace').splitlines() if l.strip()]
                    line += f'  [{logs[-1].stem}: {tail[-1][:60] if tail else "started"}]'
                break
        else:
            if done:
                last = max((s for s in stages.values() if s.get('end_time')), key=lambda s: s['end_time'])
                line += f'  idle since {last["end_time"][11:19]} UTC'
        print(line)


if __name__ == '__main__':
    main()

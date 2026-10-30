import { useEffect, useState } from 'react';
import { getJobs, type JobStatus } from './api';

const minutes = (seconds: number) => `${Math.floor(seconds / 60)}m ${String(Math.floor(seconds % 60)).padStart(2, '0')}s`;
const describe: Record<JobStatus['state'], string> = {
  running: 'running', failed: 'failed', stopped: 'waiting for GPU stages', done: 'done', pending: 'pending',
};

/** Pipeline job progress from /api/jobs; polls only while the section is open. */
export function JobsPanel() {
  const [open, setOpen] = useState(false);
  const [jobs, setJobs] = useState<JobStatus[]>();
  const [error, setError] = useState('');
  useEffect(() => {
    if (!open) return;
    const abort = new AbortController();
    const poll = () => getJobs(abort.signal).then(j => { setJobs(j); setError(''); })
      .catch(cause => { if (!abort.signal.aborted) setError(String(cause.message ?? cause)); });
    void poll();
    const timer = setInterval(poll, 10_000);
    return () => { abort.abort(); clearInterval(timer); };
  }, [open]);
  const done = jobs?.filter(j => j.state === 'done').length ?? 0;
  return <details className="jobs" onToggle={e => setOpen((e.target as HTMLDetailsElement).open)}>
    <summary>Pipeline jobs{jobs ? ` · ${done}/${jobs.length} done` : ''}</summary>
    {error && <p className="jobs-error">{error}</p>}
    {jobs && !jobs.length && <p>No jobs yet.</p>}
    {jobs && jobs.length > 0 && <ul aria-label="Pipeline jobs">
      {jobs.map(job => <li key={job.name} className={`job-${job.state}`}>
        <span className="job-name">{job.name}</span>
        <progress max={job.stages_total} value={job.stages_done} aria-label={`${job.name} stages done`} />
        <span className="job-state">{job.current
          ? `${job.current.status} ${job.current.name}${job.current.gpu ? ' (GPU)' : ''}${job.current.elapsed_seconds != null ? ` · ${minutes(job.current.elapsed_seconds)}` : ''}`
          : `${job.stages_done}/${job.stages_total} · ${describe[job.state]}`}</span>
      </li>)}
    </ul>}
  </details>;
}

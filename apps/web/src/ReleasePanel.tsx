import { apiUrl, type DatasetDetail, type Manifest, type Release } from './api';
import { captureRange, formatBytes } from './format';
// Matches pipeline/semantics.py COLOURS; unknown splats keep their own colour in greyscale.
const CLASS_LEGEND: [string, string][] = [['Building', 'rgb(220 80 60)'], ['Trees', 'rgb(60 170 70)'],
  ['Grass', 'rgb(150 200 110)'], ['Road', 'rgb(110 110 120)'], ['Ground', 'rgb(190 180 150)'],
  ['Water', 'rgb(60 120 220)'], ['Unknown', 'linear-gradient(90deg, #444, #ccc)']];
export function ReleasePanel({ dataset, release, manifest, busy, onChange }: {
  dataset: DatasetDetail; release?: Release; manifest?: Manifest; busy: boolean; onChange: (release: Release) => void;
}) {
  const q = release?.quality;
  return <section aria-label="Release quality">
    <h2>{dataset.title}</h2>
    <label htmlFor="release">Release · {dataset.dataset_id.startsWith('skyfall-') ? 'prepared Skyfall scene' : 'historical 2015'}</label>
    <select id="release" value={release?.release_id ?? ''} disabled={busy} onChange={e => {
      const chosen = dataset.releases.find(r => r.release_id === e.target.value); if (chosen) onChange(chosen);
    }}>
      {dataset.releases.map(r => <option key={r.release_id} value={r.release_id}>{r.title}{r.release_id === dataset.active_release_id ? ' · active' : ''}</option>)}
    </select>
    {release && <>
      <p className="capture">{captureRange(release.capture_dates)}</p>
      <p>{q?.splats.toLocaleString()} splats · {q?.tiles} tiles · {formatBytes(release.bytes)}</p>
    </>}
    {q && <dl>
      <dt>Registered lidar MAE</dt><dd>{q.registered_lidar_mae_all_m == null ? 'Unmeasured' : `${q.registered_lidar_mae_all_m.toFixed(2)} m`}</dd>
      <dt>Ground</dt><dd>{q.registered_lidar_mae_ground_m == null ? 'Unmeasured' : `${q.registered_lidar_mae_ground_m.toFixed(2)} m`}</dd>
      <dt>Buildings</dt><dd>{q.registered_lidar_mae_buildings_m == null ? 'Unmeasured' : `${q.registered_lidar_mae_buildings_m.toFixed(2)} m`}</dd>
      <dt>Vegetation</dt><dd>{q.registered_lidar_mae_vegetation_m == null ? 'Unmeasured' : `${q.registered_lidar_mae_vegetation_m.toFixed(2)} m`}</dd>
      <dt>Holdout PSNR</dt><dd>{q.holdout_psnr_db == null ? 'Unavailable' : `${q.holdout_psnr_db.toFixed(2)} dB`}</dd>
      <dt>Horizontal offset</dt><dd>{q.horizontal_offset_m == null ? 'Unmeasured' : `${q.horizontal_offset_m.toFixed(2)} m`}</dd>
    </dl>}
    {manifest && <>
      {manifest.source_label?.startsWith('Semantic classes') && <ul className="class-legend" aria-label="Class colours">
        {CLASS_LEGEND.map(([label, colour]) => <li key={label}><span style={{ background: colour }} />{label}</li>)}
      </ul>}
      <p>{manifest.description}</p>
      <details><summary>Limitations &amp; evidence</summary><p>{manifest.limitations}</p>
        <ul>{manifest.evidence.map(path => <li key={path}><a href={apiUrl(`/releases/${manifest.release_id}/${path}`)} target="_blank" rel="noreferrer">{path.split('/').at(-1)}</a></li>)}</ul>
      </details>
    </>}
  </section>;
}

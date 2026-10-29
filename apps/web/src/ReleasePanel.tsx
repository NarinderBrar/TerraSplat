import { apiUrl, type DatasetDetail, type Manifest, type Release } from './api';
import { captureRange, formatBytes } from './format';
export function ReleasePanel({ dataset, release, manifest, busy, onChange }: {
  dataset: DatasetDetail; release?: Release; manifest?: Manifest; busy: boolean; onChange: (release: Release) => void;
}) {
  const q = release?.quality;
  return <section aria-label="Release quality">
    <h2>{dataset.title}</h2>
    <label htmlFor="release">Release · historical 2015</label>
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
      <dt>Registered lidar MAE</dt><dd>{q.registered_lidar_mae_all_m.toFixed(2)} m</dd>
      <dt>Ground</dt><dd>{q.registered_lidar_mae_ground_m.toFixed(2)} m</dd>
      <dt>Buildings</dt><dd>{q.registered_lidar_mae_buildings_m.toFixed(2)} m</dd>
      <dt>Vegetation</dt><dd>{q.registered_lidar_mae_vegetation_m.toFixed(2)} m</dd>
      <dt>Holdout PSNR</dt><dd>{q.holdout_psnr_db.toFixed(2)} dB</dd>
      <dt>Horizontal offset</dt><dd>{q.horizontal_offset_m.toFixed(2)} m</dd>
    </dl>}
    {manifest && <>
      <p>{manifest.description}</p>
      <details><summary>Limitations &amp; evidence</summary><p>{manifest.limitations}</p>
        <ul>{manifest.evidence.map(path => <li key={path}><a href={apiUrl(`/releases/${manifest.release_id}/${path}`)} target="_blank" rel="noreferrer">{path.split('/').at(-1)}</a></li>)}</ul>
      </details>
    </>}
  </section>;
}

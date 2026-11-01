import { useEffect, useState } from 'react';
import { getSegments, type Manifest, type SegmentLayer } from './api';

const NAMES: Record<string, string> = { building: 'Buildings', road: 'Roads', vegetation: 'Trees', grass: 'Grass',
  water: 'Water', ground: 'Ground', unknown: 'Unlabelled' };

/** Class list of a release's segment layer; chosen classes are highlighted (bit per class code), none chosen is the normal view. */
export function SegmentPanel({ manifest, available, onChange }: {
  manifest?: Manifest; available: boolean; onChange: (mode: boolean, mask: number) => void;
}) {
  const [layer, setLayer] = useState<SegmentLayer>();
  const [mask, setMask] = useState(0);
  const [error, setError] = useState('');
  const segments = manifest?.segments;
  useEffect(() => {
    setLayer(undefined); setError('');
    if (!manifest || !segments) return;
    const abort = new AbortController();
    getSegments(manifest.release_id, segments.path, abort.signal).then(setLayer)
      .catch(cause => { if (!abort.signal.aborted) setError(String(cause.message ?? cause)); });
    return () => abort.abort();
  }, [manifest, segments]);
  if (!segments) return null;
  const update = (next: number) => { setMask(next); onChange(next !== 0, next); };
  // Unlabelled splats cannot be highlighted.
  const classes = layer?.classes.filter(c => c.code !== 0 && (layer.leaf_splats_by_class[c.name] ?? 0) > 0)
    .sort((a, b) => a.code - b.code) ?? [];
  return <section className="segments" aria-label="Segments">
    <h3 className="segments-title">Classes</h3>
    {!available && <p className="segments-note">Unavailable: this browser's Cesium splat shader could not be extended.</p>}
    {error && <p className="segments-note">{error}</p>}
    {available && layer && <>
      <p className="segments-note">{mask ? 'Selected classes glow; everything else keeps its colours.' : 'Choose classes to highlight.'}</p>
      <ul aria-label="Segment classes">{classes.map(c => {
        const bit = 1 << c.code, on = (mask & bit) !== 0;
        return <li key={c.code}><button aria-pressed={on} onClick={() => update(mask ^ bit)}>
          <span style={{ background: `rgb(${c.colour.join(' ')})` }} />{NAMES[c.name] ?? c.name}
          <small>{(layer.leaf_splats_by_class[c.name] ?? 0).toLocaleString()}</small></button></li>;
      })}</ul>
      {mask !== 0 && <button className="segments-clear" onClick={() => update(0)}>Clear</button>}
      <p className="segments-note">{layer.label}</p>
    </>}
  </section>;
}

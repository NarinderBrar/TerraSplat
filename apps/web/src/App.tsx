import { useEffect, useRef, useState } from 'react';
import { getDatasets, getDataset, getFootprints, getManifest, type Dataset, type DatasetDetail, type Footprint, type Manifest, type Release } from './api';
import { footprintTooltip, formatBytes } from './format';
import { Globe, isSkyfall, type MapStats, type Layers } from './globe';
import { ReleasePanel } from './ReleasePanel';

export function App() {
  const host = useRef<HTMLDivElement>(null);
  const globe = useRef<Globe | null>(null);
  const datasets = useRef<Dataset[]>([]);
  const selection = useRef(0);
  const selectedDataset = useRef('');
  const [footprints, setFootprints] = useState<Footprint[]>([]);
  const [showFootprints, setShowFootprints] = useState(false);
  const [panel, setPanel] = useState(true);
  const [dataset, setDataset] = useState<DatasetDetail>();
  const [release, setRelease] = useState<Release>();
  const [manifest, setManifest] = useState<Manifest>();
  const [ready, setReady] = useState(false);
  const [busy, setBusy] = useState(false);
  const [debug, setDebug] = useState(false);
  const [reveal, setReveal] = useState(() => !window.matchMedia?.('(prefers-reduced-motion: reduce)').matches);
  const [layers, setLayers] = useState<Layers>({ pipeline: true, skyfall: true, overlap: 'skyfall' });
  const [error, setError] = useState('');
  const [tooltip, setTooltip] = useState({ text: '', x: 0, y: 0 });
  const [stats, setStats] = useState<MapStats>({ loaded: 0, visible: 0, bytes: 0, heading: 0, height: 22_000_000 });
  const fail = (cause: unknown) => setError(cause instanceof Error ? cause.message :
    cause && typeof cause === 'object' && 'message' in cause ? String(cause.message) : String(cause));

  useEffect(() => {
    const abort = new AbortController();
    let map: Globe;
    async function select(f: Footprint, fly: boolean) {
      if (fly) {
        setLayers(previous => {
          const next = { ...previous, overlap: isSkyfall(f.properties.dataset_id) ? 'skyfall' as const : 'pipeline' as const };
          map.setLayers(next); return next;
        });
        map.flyTo(f); setPanel(true);
      }
      if (selectedDataset.current === f.properties.dataset_id) return;
      selectedDataset.current = f.properties.dataset_id;
      const current = ++selection.current;
      setManifest(undefined); setRelease(undefined); setBusy(true);
      try {
        const d = await getDataset(f.properties.dataset_id, abort.signal);
        const r = d.active_release;
        const m = r ? await getManifest(r.release_id, abort.signal) : undefined;
        if (current !== selection.current || abort.signal.aborted) return;
        setDataset(d); setRelease(r ?? undefined); setManifest(m);
      } catch (cause) { if (!abort.signal.aborted) { selectedDataset.current = ''; fail(cause); } }
      finally { if (current === selection.current && !abort.signal.aborted) setBusy(false); }
    }
    try {
      map = new Globe(host.current!, setStats, (text, x, y) => setTooltip({ text, x, y }), f => { void select(f, true); }, fail);
      map.onSelectInView = f => { void select(f, false); };
      globe.current = map;
      void (async () => {
        const [ds, fs] = await Promise.all([getDatasets(abort.signal), getFootprints(abort.signal)]);
        if (abort.signal.aborted) return;
        datasets.current = ds; setFootprints(fs); map.setFootprints(fs);
        await map.loadIndexes(ds, abort.signal);
        if (!abort.signal.aborted) setReady(true);
      })().catch(cause => { if (!abort.signal.aborted) fail(cause); });
    } catch (cause) { fail(cause); }
    return () => { abort.abort(); ++selection.current; map?.destroy(); globe.current = null; };
  }, []);

  async function switchRelease(r: Release) {
    if (!dataset || !globe.current) return;
    const current = ++selection.current;
    setBusy(true); setError('');
    try {
      const m = await getManifest(r.release_id);
      await globe.current.switchRelease(dataset, r, datasets.current);
      if (selection.current !== current) return;
      setRelease(r); setManifest(m);
    } catch (cause) { fail(cause); }
    finally { if (selection.current === current) setBusy(false); }
  }
  function changeLayers(next: Layers) {
    setLayers(next); globe.current?.setLayers(next);
    // If the selected block is now hidden behind an overlapping block of the other kind, select that block instead.
    const selected = selectedDataset.current, other = globe.current?.overlapping(selected) ?? [];
    const showPipeline = next.pipeline && (!next.skyfall || next.overlap === 'pipeline');
    if (other.length && isSkyfall(selected) === showPipeline) {
      const f = footprints.find(f => f.properties.dataset_id === other[0]);
      if (f) globe.current?.onSelectInView?.(f);
    }
  }
  function choose(f: Footprint) {
    if (isSkyfall(f.properties.dataset_id)) changeLayers({ ...layers, skyfall: true, overlap: 'skyfall' });
    else changeLayers({ ...layers, pipeline: true, overlap: 'pipeline' });
    // The map and keyboard list share the same selection path.
    globe.current?.onSelectInView?.(f);
    globe.current?.flyTo(f); setPanel(true);
  }
  return <>
    <div ref={host} className="globe" aria-label="TerraSplat globe" />
    <header className="brand"><span className="brand-mark">T</span><div><h1>TerraSplat</h1><span>Earth, reconstructed.</span></div>
      <button aria-label={panel ? 'Collapse panel' : 'Expand panel'} aria-expanded={panel} onClick={() => setPanel(!panel)}>{panel ? '‹' : '›'}</button>
    </header>
    <aside className={`panel ${panel ? '' : 'collapsed'}`} aria-label="Datasets and releases">
      <p className="eyebrow">EXPLORE THE ARCHIVE</p>
      <p>Find a footprint. Step into a place.</p>
      <fieldset className="layers"><legend>Layers</legend>
        <label><input type="checkbox" checked={layers.pipeline} onChange={e => changeLayers({ ...layers, pipeline: e.target.checked })} /> Pipeline 2015 (19 photos)</label>
        <label><input type="checkbox" checked={layers.skyfall} onChange={e => changeLayers({ ...layers, skyfall: e.target.checked })} /> Skyfall scenes</label>
        <label htmlFor="overlap">JAX_068 overlap</label>
        <select id="overlap" value={layers.overlap} onChange={e => changeLayers({ ...layers, overlap: e.target.value as Layers['overlap'] })}>
          <option value="skyfall">Skyfall</option><option value="pipeline">Pipeline 2015</option>
        </select>
      </fieldset>
      <button className="footprints-toggle" aria-pressed={showFootprints} disabled={!ready} onClick={() => {
        const next = !showFootprints; setShowFootprints(next); globe.current?.showFootprints(next); setTooltip({ text: '', x: 0, y: 0 });
      }}>◇ &nbsp; 3D Footprints</button>
      {showFootprints && <ul className="footprint-list">{footprints.map(f => <li key={f.properties.release_id}>
        <button title={footprintTooltip(f)} onClick={() => choose(f)}>{f.properties.title}<span>{f.properties.dataset_id.startsWith('skyfall-') ? `${f.properties.source_label ?? 'Skyfall'} · visual registration` : 'Pipeline 2015 · known anchor'} · fly to block ↗</span></button>
      </li>)}</ul>}
      {!dataset && <p className="hint">Start with the whole Earth. Enable footprints to discover available 3D blocks, or pan and zoom to explore.</p>}
      {dataset && <ReleasePanel dataset={dataset} release={release} manifest={manifest} busy={busy} onChange={r => { void switchRelease(r); }} />}
      <label className="debug-toggle"><input type="checkbox" checked={reveal} onChange={e => { setReveal(e.target.checked); globe.current?.setReveal(e.target.checked); }} /> Reveal effects</label>
      <label className="debug-toggle"><input type="checkbox" checked={debug} onChange={e => { setDebug(e.target.checked); globe.current?.setDebug(e.target.checked); }} /> Tile diagnostics</label>
      {debug && <output data-testid="tile-stats">Loaded {stats.loaded} · visible {stats.visible} · {formatBytes(stats.bytes)} in memory</output>}
      <p role="status">{busy ? 'Loading release…' : ready ? 'Ready · tiles stream as you explore' : 'Loading globe and catalogue…'}</p>
      {error && <p role="alert">{error}</p>}
    </aside>
    <nav className="map-controls" aria-label="Camera controls">
      <button aria-label="Reset north" title="Reset heading to north" onClick={() => globe.current?.north()}><span style={{ display: 'inline-block', transform: `rotate(${-stats.heading}deg)` }}>↑</span><small>N</small></button>
      <button aria-label="Top-down view" title="Top-down view" onClick={() => globe.current?.north(true)}>⊤</button>
      <button aria-label="Zoom in" onClick={() => globe.current?.zoom(0.5)}>+</button>
      <button aria-label="Zoom out" onClick={() => globe.current?.zoom(2)}>−</button>
    </nav>
    <div className="map-hint">Drag to pan · scroll to zoom · right-drag to orbit</div>
    {tooltip.text && <div role="tooltip" style={{ left: Math.min(tooltip.x + 18, window.innerWidth - 300), top: Math.min(tooltip.y + 18, window.innerHeight - 100) }}>{tooltip.text}</div>}
  </>;
}

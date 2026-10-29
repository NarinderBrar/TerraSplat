import type { Footprint, Release } from './api';
export function captureRange(dates: Release['capture_dates']) {
  const all = [...dates.train, ...dates.holdout].sort();
  return all.length && /^\d{4}-/.test(all[0]) ? `${all[0].slice(0, 10)} – ${all.at(-1)!.slice(0, 10)}` : 'Capture dates unavailable';
}
export function footprintTooltip(f: Footprint) {
  const p = f.properties;
  return `${p.title}\n${p.source_label ?? 'TerraSplat raw-imagery pipeline · historical 2015'}\n${p.accuracy_label ?? 'Known anchor; no ground control'}${p.headline_mae_m == null ? '' : `\nRegistered lidar MAE ${p.headline_mae_m.toFixed(2)} m`}`;
}
export function footprintCentre(f: Footprint): [number, number] {
  const ring = f.geometry.coordinates[0];
  const points = ring.slice(0, -1);
  return [0, 1].map(axis => points.reduce((sum, p) => sum + p[axis], 0) / points.length) as [number, number];
}
export const formatBytes = (bytes: number) => `${(bytes / 1_000_000).toFixed(2)} MB`;

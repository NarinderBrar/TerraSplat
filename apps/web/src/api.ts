export interface Quality {
  registered_lidar_mae_all_m: number;
  registered_lidar_mae_ground_m: number;
  registered_lidar_mae_buildings_m: number;
  registered_lidar_mae_vegetation_m: number;
  holdout_psnr_db: number;
  horizontal_offset_m: number;
  splats: number;
  tiles: number;
}
export interface Release {
  release_id: string; title: string; tileset_url: string; bytes: number;
  capture_dates: { train: string[]; holdout: string[] }; quality: Quality;
}
export interface Dataset {
  dataset_id: string; title: string; region: string;
  active_release_id: string | null; active_release: Release | null;
}
export interface DatasetDetail extends Dataset { releases: Release[] }
export interface Footprint {
  type: 'Feature'; geometry: { type: 'Polygon'; coordinates: number[][][] };
  properties: { dataset_id: string; release_id: string; title: string;
    capture_date_start: string; capture_date_end: string; headline_mae_m: number; tileset_url: string };
}
export interface Manifest {
  release_id: string; dataset_id: string; title: string; description: string;
  anchor: { latitude: number; longitude: number; ellipsoidal_height_m: number };
  footprint: Footprint['geometry']; limitations: string; evidence: string[];
}
const base = (import.meta.env.VITE_API_BASE || '').replace(/\/$/, '');
export const apiUrl = (path: string) => `${base}${path}`;
function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Invalid API object');
  return value as Record<string, unknown>;
}
function field(value: unknown, type: 'string' | 'number') {
  if (typeof value !== type || (type === 'number' && !Number.isFinite(value))) throw new Error(`Invalid API ${type}`);
}
function strings(value: unknown) {
  if (!Array.isArray(value)) throw new Error('Invalid API list');
  value.forEach(item => field(item, 'string'));
}
function release(value: unknown) {
  const r = object(value);
  for (const key of ['release_id', 'title', 'tileset_url']) field(r[key], 'string');
  field(r.bytes, 'number');
  const dates = object(r.capture_dates); strings(dates.train); strings(dates.holdout);
  const q = object(r.quality);
  for (const key of ['registered_lidar_mae_all_m', 'registered_lidar_mae_ground_m',
    'registered_lidar_mae_buildings_m', 'registered_lidar_mae_vegetation_m',
    'holdout_psnr_db', 'horizontal_offset_m', 'splats', 'tiles']) field(q[key], 'number');
  return value as Release;
}
export function parseDatasets(value: unknown): Dataset[] {
  if (!Array.isArray(value)) throw new Error('Invalid dataset list');
  for (const item of value) {
    const d = object(item);
    for (const key of ['dataset_id', 'title', 'region']) field(d[key], 'string');
    if (d.active_release_id !== null) field(d.active_release_id, 'string');
    if (d.active_release !== null) release(d.active_release);
  }
  return value as Dataset[];
}
export function parseFootprints(value: unknown): Footprint[] {
  const collection = object(value);
  if (collection.type !== 'FeatureCollection' || !Array.isArray(collection.features)) throw new Error('Invalid GeoJSON');
  for (const feature of collection.features) {
    const f = object(feature), g = object(f.geometry), p = object(f.properties);
    if (f.type !== 'Feature' || g.type !== 'Polygon' || !Array.isArray(g.coordinates) || !g.coordinates.length) throw new Error('Invalid footprint');
    for (const ring of g.coordinates) {
      if (!Array.isArray(ring) || ring.length < 4) throw new Error('Invalid polygon ring');
      for (const point of ring) {
        if (!Array.isArray(point) || point.length < 2) throw new Error('Invalid coordinate');
        point.forEach(n => field(n, 'number'));
        if (Math.abs(point[0]) > 180 || Math.abs(point[1]) > 90) throw new Error('Invalid longitude/latitude');
      }
    }
    for (const key of ['dataset_id', 'release_id', 'title', 'capture_date_start', 'capture_date_end', 'tileset_url']) field(p[key], 'string');
    field(p.headline_mae_m, 'number');
  }
  return collection.features as Footprint[];
}
export async function getJson(path: string, signal?: AbortSignal): Promise<unknown> {
  const response = await fetch(apiUrl(path), { signal });
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.json();
}
export const getDatasets = async (signal?: AbortSignal) => parseDatasets(await getJson('/api/datasets', signal));
export const getFootprints = async (signal?: AbortSignal) => parseFootprints(await getJson('/api/footprints', signal));
export async function getDataset(id: string, signal?: AbortSignal): Promise<DatasetDetail> {
  const data = object(await getJson(`/api/datasets/${encodeURIComponent(id)}`, signal));
  parseDatasets([data]);
  if (!Array.isArray(data.releases)) throw new Error('Invalid releases');
  data.releases.forEach(release);
  return data as unknown as DatasetDetail;
}
export async function getManifest(id: string, signal?: AbortSignal): Promise<Manifest> {
  const data = object(await getJson(`/api/releases/${encodeURIComponent(id)}`, signal));
  for (const key of ['release_id', 'dataset_id', 'title', 'description', 'limitations']) field(data[key], 'string');
  const anchor = object(data.anchor);
  for (const key of ['latitude', 'longitude', 'ellipsoidal_height_m']) field(anchor[key], 'number');
  strings(data.evidence);
  return data as unknown as Manifest;
}

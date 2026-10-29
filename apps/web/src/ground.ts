import type { Footprint } from './api';
import { footprintCentre } from './format';

/** A release's display ground: its centre and the ellipsoidal height of the model's ground. */
export interface GroundPatch { lon: number; lat: number; height: number }

/** Full ground height within INNER metres of a footprint centre, easing back to the ellipsoid by OUTER. */
export const INNER = 2_000, OUTER = 8_000;

export function groundPatches(features: Footprint[]): GroundPatch[] {
  return features.flatMap(f => {
    const height = f.properties.ground_height_m;
    if (height == null) return [];
    const [lon, lat] = footprintCentre(f);
    return [{ lon, lat, height }];
  });
}

function distance(lon: number, lat: number, p: GroundPatch) {
  const dx = (lon - p.lon) * 111_320 * Math.cos(p.lat * Math.PI / 180), dy = (lat - p.lat) * 110_540;
  return Math.hypot(dx, dy);
}

export function groundHeight(lon: number, lat: number, patches: GroundPatch[]) {
  let weighted = 0, total = 0;
  for (const p of patches) {
    const d = distance(lon, lat, p);
    if (d >= OUTER) continue;
    const t = d <= INNER ? 0 : (d - INNER) / (OUTER - INNER);
    const w = 1 - t * t * (3 - 2 * t);
    weighted += w * p.height; total += w;
  }
  // Inside overlapping patches use their weighted mean; in the blend zone ease towards 0.
  return total ? weighted / Math.max(1, total) : 0;
}

/** Row-major heights, north row first, as Cesium's CustomHeightmapTerrainProvider expects. */
export function heightmap(west: number, south: number, east: number, north: number, size: number, patches: GroundPatch[]) {
  const heights = new Float32Array(size * size);
  const near = patches.filter(p => {
    const lon = Math.min(Math.max(p.lon, west), east), lat = Math.min(Math.max(p.lat, south), north);
    return distance(lon, lat, p) < OUTER;
  });
  if (!near.length) return heights;
  for (let row = 0; row < size; row++) {
    const lat = north - (north - south) * row / (size - 1);
    for (let col = 0; col < size; col++) {
      heights[row * size + col] = groundHeight(west + (east - west) * col / (size - 1), lat, near);
    }
  }
  return heights;
}

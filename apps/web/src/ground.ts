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
  let nearest: GroundPatch | undefined, nearestDistance = OUTER;
  for (const p of patches) {
    const d = distance(lon, lat, p);
    if (d < nearestDistance) { nearest = p; nearestDistance = d; }
  }
  if (!nearest) return 0;
  const t = Math.max(0, (nearestDistance - INNER) / (OUTER - INNER));
  return nearest.height * (1 - t * t * (3 - 2 * t));
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

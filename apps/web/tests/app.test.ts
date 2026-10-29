import { describe, expect, it, vi, afterEach } from 'vitest';
import { CameraEventType, KeyboardEventModifier } from 'cesium';
import * as C from 'cesium';
import { controlMapping } from '../src/controls';
import { apiUrl, getDatasets, parseDatasets, parseFootprints } from '../src/api';
import { captureRange, footprintCentre, footprintTooltip, formatBytes } from '../src/format';
import { groundHeight, groundPatches, heightmap } from '../src/ground';
import datasets from './fixtures/datasets.json';
import footprints from './fixtures/footprints.json';
afterEach(() => vi.unstubAllGlobals());
describe('Google-style controls', () => {
  it('grabs the globe with left drag and reserves right drag for orbit', () => {
    const mapping = controlMapping(C);
    expect(mapping.rotateEventTypes).toEqual([CameraEventType.LEFT_DRAG]);
    expect(mapping.zoomEventTypes).toEqual([CameraEventType.WHEEL, CameraEventType.PINCH]);
    expect(mapping.tiltEventTypes).toEqual([CameraEventType.RIGHT_DRAG, CameraEventType.PINCH,
      { eventType: CameraEventType.LEFT_DRAG, modifier: KeyboardEventModifier.CTRL }]);
    expect(mapping.lookEventTypes).toEqual([]);
    expect(mapping.minimumZoomDistance).toBeGreaterThan(0);
    expect(mapping.maximumZoomDistance).toBeGreaterThan(22_000_000);
  });
});
describe('API contract', () => {
  it('parses published dataset and GeoJSON response fixtures', () => {
    expect(parseDatasets(datasets)[0].active_release?.quality.splats).toBe(61857);
    expect(parseFootprints(footprints)[0].properties.release_id).toBe(datasets[0].active_release_id);
  });
  it('rejects malformed metadata rather than rendering undefined values', () => {
    expect(() => parseDatasets({ datasets })).toThrow();
    expect(() => parseDatasets([{ ...datasets[0], active_release: { ...datasets[0].active_release, quality: {} } }])).toThrow();
    expect(() => parseFootprints({ ...footprints, features: [{ ...footprints.features[0], geometry: { type: 'Point' } }] })).toThrow();
    expect(() => parseFootprints({ ...footprints, features: [{ ...footprints.features[0], properties: {} }] })).toThrow();
  });
  it('reports HTTP failures and parses a successful fetch', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify(datasets))));
    expect(await getDatasets()).toEqual(datasets);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('', { status: 503 })));
    await expect(getDatasets()).rejects.toThrow('HTTP 503');
    expect(apiUrl('/api/datasets')).toBe('/api/datasets');
  });
});
describe('footprint presentation', () => {
  it('uses the published bounds and capture dates with units', () => {
    const feature = parseFootprints(footprints)[0];
    const [lon, lat] = footprintCentre(feature);
    expect(lon).toBeCloseTo(-81.6638, 3); expect(lat).toBeCloseTo(30.3488, 3);
    expect(footprintTooltip(feature)).toContain('Registered lidar MAE 2.82 m');
    expect(footprintTooltip(feature)).toContain('historical 2015');
    expect(captureRange(datasets[0].active_release.capture_dates)).toBe('2015-04-19 – 2015-06-15');
    expect(captureRange({ train: [], holdout: [] })).toBe('Capture dates unavailable');
    expect(formatBytes(1591466)).toBe('1.59 MB');
  });
});

describe('Cesium 1.146 regional index compatibility', () => {
  it('declares the outer splat profile and resolves external URLs without mutating source data', async () => {
    const { regionalIndexUrl } = await import('../src/regionalIndex');
    const source = { asset: { version: '1.1' }, root: {
      transform: [1, 2, 3], boundingVolume: { box: [1, 2, 3] }, refine: 'ADD',
      children: [{ content: { uri: '../../jax068-2015-5v-r1/tileset.json' } }],
    } };
    const url = regionalIndexUrl(source, 'http://localhost:8001/releases/index/jacksonville/tileset.json');
    const result = JSON.parse(decodeURIComponent(url.split(',')[1]));
    expect(result.extensions['3DTILES_content_gltf'].extensionsRequired).toContain('KHR_gaussian_splatting');
    expect(result.root.children[0].content.uri).toBe('http://localhost:8001/releases/jax068-2015-5v-r1/tileset.json');
    expect(result.root.transform).toEqual(source.root.transform);
    expect(source.root.children[0].content.uri).toBe('../../jax068-2015-5v-r1/tileset.json');
    expect(() => regionalIndexUrl({}, 'http://localhost/')).toThrow('Invalid regional tileset');
  });
});
describe('display ground terrain', () => {
  const base = parseFootprints(footprints)[0];
  const feature = { ...base, properties: { ...base.properties, ground_height_m: -20.75 } };
  const [lon, lat] = footprintCentre(feature);
  it('uses the release ground near the block and the ellipsoid far away', () => {
    const patches = groundPatches([feature]);
    expect(groundHeight(lon, lat, patches)).toBeCloseTo(-20.75, 6);
    expect(groundHeight(lon + 0.015, lat, patches)).toBeCloseTo(-20.75, 6); // ~1.4 km east
    expect(groundHeight(lon + 0.2, lat, patches)).toBe(0); // ~19 km east
    const mid = groundHeight(lon + 0.05, lat, patches); // ~4.8 km: easing zone
    expect(mid).toBeLessThan(0);
    expect(mid).toBeGreaterThan(-20.75);
  });
  it('skips releases without a ground height and fills heightmaps north row first', () => {
    expect(groundPatches(parseFootprints(footprints))).toEqual([]);
    const patches = groundPatches([feature]);
    const h = heightmap(lon - 0.001, lat - 0.001, lon + 0.001, lat + 0.001, 4, patches);
    expect([...h].every(v => Math.abs(v + 20.75) < 1e-4)).toBe(true);
    expect([...heightmap(0, 0, 1, 1, 4, patches)].every(v => v === 0)).toBe(true);
  });
});

it('keeps neighbouring terrain patches at their own model ground', () => {
  const patches = [{ lon: -81.66, lat: 30.35, height: -17 }, { lon: -81.65, lat: 30.35, height: -25 }];
  expect(groundHeight(-81.66, 30.35, patches)).toBe(-17);
  expect(groundHeight(-81.65, 30.35, patches)).toBe(-25);
});

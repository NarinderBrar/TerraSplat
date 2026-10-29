import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { gunzipSync } from 'node:zlib';
import assert from 'node:assert/strict';
import createSpzModule from '@adobe/spz';
import { Cartesian3, Matrix4, Transforms } from 'cesium';
import { C0, baseExtension, compressionExtension, packGlb, splatGltf, tilesetExtensions } from './lib/splat-glb.mjs';

const output = new URL('../public/phase0/', import.meta.url);
// glTF is right-handed Y-up: (east, north, up) -> (east, up, -north).
const samples = [
  { name: 'origin', enu: [0, 0, 0], rgb: [0.8, 0.8, 0.8], scale: [0.35, 0.35, 0.35], rotation: [0, 0, 0, 1], opacity: 0.9 },
  { name: 'east', enu: [10, 0, 0], rgb: [1, 0.05, 0.05], scale: [1.5, 0.3, 0.3], rotation: [0, 0, 0, 1], opacity: 0.9 },
  { name: 'north', enu: [0, 10, 0], rgb: [0.05, 1, 0.05], scale: [0.3, 0.3, 1.5], rotation: [0, 0, 0, 1], opacity: 0.9 },
  { name: 'up', enu: [0, 0, 10], rgb: [0.05, 0.05, 1], scale: [0.3, 1.5, 0.3], rotation: [0, 0, 0, 1], opacity: 0.9 },
  { name: 'rotated', enu: [-10, 0, 0], rgb: [1, 0.65, 0.05], scale: [1.5, 0.3, 0.3], rotation: [0, 0, Math.sin(Math.PI / 8), Math.cos(Math.PI / 8)], opacity: 0.5 },
];
for (const sample of samples) sample.gltf = [sample.enu[0], sample.enu[2], -sample.enu[1]];
const flat = (fn) => new Float32Array(samples.flatMap(fn));
const cloud = {
  numPoints: samples.length, shDegree: 0, antialiased: false, extensions: [],
  positions: flat(s => s.gltf), scales: flat(s => s.scale.map(Math.log)),
  rotations: flat(s => s.rotation), alphas: flat(s => Math.log(s.opacity / (1 - s.opacity))),
  colors: flat(s => s.rgb.map(c => (c - 0.5) / C0)), sh: new Float32Array(),
};
const spz = await createSpzModule();
// Explicitly request v2; the encoder default is a different codec profile.
const payload = spz.saveSpzToBuffer(cloud, { version: 2, from: spz.CoordinateSystem.UNSPECIFIED, sh1Bits: 5, shRestBits: 4 });
const header = gunzipSync(payload);
assert.equal(header.readUInt32LE(0), 0x5053474e);
assert.equal(header.readUInt32LE(4), 2);
assert.equal(header.readUInt32LE(8), samples.length);
const decoded = spz.loadSpzFromBuffer(payload, { to: spz.CoordinateSystem.UNSPECIFIED });
const maxError = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])));
const roundtrip = {};
for (const [field, tolerance] of [['positions', 0.001], ['scales', 0.04], ['colors', 0.08], ['alphas', 0.04]]) {
  const error = maxError([...cloud[field]], decoded[field]);
  assert.ok(error < tolerance, `${field} quantization error ${error}`);
  roundtrip[field] = { maxAbsoluteError: error, tolerance };
}
for (let i = 0; i < samples.length; i++) {
  const q = decoded.rotations.slice(i * 4, i * 4 + 4);
  const dot = Math.abs(q.reduce((sum, v, j) => sum + v * samples[i].rotation[j], 0));
  assert.ok(dot > 0.999, `quaternion ${i} changed orientation`);
}
const mins = [0, 1, 2].map(i => Math.min(...samples.map(s => s.gltf[i])));
const maxs = [0, 1, 2].map(i => Math.max(...samples.map(s => s.gltf[i])));
const gltf = splatGltf({ count: samples.length, byteLength: payload.length, min: mins, max: maxs,
  generator: 'TerraSplat Phase 0 / @adobe/spz 0.2.3' });
const anchor = { longitude: -81.66376054127814, latitude: 30.34881972282938, ellipsoidalHeight: 100 };
const transform = Matrix4.toArray(Transforms.eastNorthUpToFixedFrame(Cartesian3.fromDegrees(anchor.longitude, anchor.latitude, anchor.ellipsoidalHeight)));
const tileset = {
  // Nonzero tileset error allows Cesium to select this leaf; the leaf has no finer LOD.
  asset: { version: '1.1' }, geometricError: 20,
  ...tilesetExtensions,
  root: { transform, boundingVolume: { sphere: [0, 0, 0, 20] }, geometricError: 0,
    refine: 'REPLACE', content: { uri: 'fixture.glb' } },
};
const fixture = { anchor, units: 'metres', gltfAxes: ['east', 'up', 'south'], samples,
  registration: 'Synthetic test anchor only; not surveyed scene registration.' };
await mkdir(output, { recursive: true });
const files = { 'fixture.glb': packGlb(gltf, payload), 'fixture.spz': payload,
  'tileset.json': JSON.stringify(tileset, null, 2) + '\n', 'fixture.json': JSON.stringify(fixture, null, 2) + '\n' };
for (const [name, data] of Object.entries(files)) await writeFile(new URL(name, output), data);
const pkg = JSON.parse(await readFile(new URL('../package.json', import.meta.url)));
const sha = data => createHash('sha256').update(data).digest('hex');
const report = {
  profile: 'CesiumJS SPZ-v2 compatibility fixture; not a general production exporter',
  dependencies: { ...pkg.dependencies }, spzHeaderVersion: 2, shDegree: 0,
  extensions: { base: baseExtension, compression: compressionExtension,
    cesiumProfileRevision: '511dbc90beacddd0085f7b6022a36da332a3b836' },
  axes: 'glTF (E,U,-N); Cesium glTF Y-up to tile Z-up; tile ENU to ECEF',
  kernel: 'ellipse', colorSpace: 'srgb_rec709_display', roundtrip,
  hashes: Object.fromEntries(Object.entries(files).map(([name, data]) => [name, sha(data)])),
  cesiumDecoderSha256: sha(await readFile(new URL('../node_modules/cesium/Build/Cesium/ThirdParty/wasm_splats_bg.wasm', import.meta.url))),
  gates: { structure: 'passed generator assertions', browserRendering: 'run npm run test:browser', realScene: 'pending Phase 1' },
};
await writeFile(new URL('compatibility.json', output), JSON.stringify(report, null, 2) + '\n');
console.log(`Generated ${samples.length} splats: SPZ v2, ${payload.length} bytes; round-trip checks passed.`);

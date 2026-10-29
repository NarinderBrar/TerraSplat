// Encode a pipeline/tiling.py plan into SPZ-v2 glb tiles and a 3D Tiles 1.1 tileset.
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { gunzipSync } from 'node:zlib';
import assert from 'node:assert/strict';
import createSpzModule from '@adobe/spz';
import { packGlb, splatGltf, tilesetExtensions } from './lib/splat-glb.mjs';

const RECORD = 14;
const [directory] = process.argv.slice(2);
if (!directory) throw new Error('Usage: node scripts/encode-tiles.mjs <tiles directory>');
const root = new URL(`${directory.replace(/\/?$/, '/')}`, `file://${process.cwd()}/`);
const sha = data => createHash('sha256').update(data).digest('hex');
const plan = JSON.parse(await readFile(new URL('tile-plan.json', root)));
assert.equal(plan.status, 'planned');
const spz = await createSpzModule();
const pkg = JSON.parse(await readFile(new URL('../package.json', import.meta.url)));
const generator = `TerraSplat tiler / @adobe/spz ${pkg.dependencies['@adobe/spz']}`;
await mkdir(new URL('tiles/', root), { recursive: true });

const sigmoid = v => 1 / (1 + Math.exp(-v));
const maxError = (a, b) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);
const report = { tiles: {}, totals: { splats: 0, glbBytes: 0 } };
const nodes = {};
for (const tile of plan.tiles) {
  const raw = await readFile(new URL(tile.payload, root));
  assert.equal(sha(raw), tile.payload_sha256, `${tile.id} payload changed after planning`);
  const records = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
  const count = records.length / RECORD;
  assert.equal(count, tile.splats);
  const field = (offset, width) => {
    const out = new Float32Array(count * width);
    for (let i = 0; i < count; i++) for (let j = 0; j < width; j++) out[i * width + j] = records[i * RECORD + offset + j];
    return out;
  };
  const cloud = { numPoints: count, shDegree: 0, antialiased: false, extensions: [],
    positions: field(0, 3), scales: field(3, 3), rotations: field(6, 4), alphas: field(10, 1),
    colors: field(11, 3), sh: new Float32Array() };
  const payload = spz.saveSpzToBuffer(cloud, { version: 2, from: spz.CoordinateSystem.UNSPECIFIED, sh1Bits: 5, shRestBits: 4 });
  const header = gunzipSync(payload);
  assert.equal(header.readUInt32LE(0), 0x5053474e);
  assert.equal(header.readUInt32LE(4), 2);
  assert.equal(header.readUInt32LE(8), count);
  const decoded = spz.loadSpzFromBuffer(payload, { to: spz.CoordinateSystem.UNSPECIFIED });
  const positionError = maxError([...cloud.positions], decoded.positions);
  assert.ok(positionError < 1e-3, `${tile.id} position quantization error ${positionError} m`);
  // Decoded centres must stay inside the tile box (glTF (e, u, -n) back to ENU).
  const [cx, cy, cz, hx, , , , hy, , , , hz] = tile.box;
  for (let i = 0; i < count; i++) {
    const [e, u, s] = decoded.positions.slice(i * 3, i * 3 + 3);
    assert.ok(Math.abs(e - cx) <= hx + 1e-3 && Math.abs(-s - cy) <= hy + 1e-3 && Math.abs(u - cz) <= hz + 1e-3,
      `${tile.id} decoded splat ${i} lies outside its box`);
  }
  const min = [0, 1, 2].map(j => Math.min(...cloud.positions.filter((_, i) => i % 3 === j)));
  const max = [0, 1, 2].map(j => Math.max(...cloud.positions.filter((_, i) => i % 3 === j)));
  const glb = packGlb(splatGltf({ count, byteLength: payload.length, min, max, generator }), payload);
  const uri = `tiles/${tile.id}.glb`;
  await writeFile(new URL(uri, root), glb);
  report.tiles[tile.id] = { splats: count, glbBytes: glb.length, glbSha256: sha(glb), positionError,
    // SPZ stores 8-bit activated opacity, so compare after the sigmoid (endpoints decode to +/-inf logits).
    opacityError: maxError([...cloud.alphas].map(sigmoid), [...decoded.alphas].map(sigmoid)), scaleError: maxError([...cloud.scales], decoded.scales) };
  report.totals.splats += count;
  report.totals.glbBytes += glb.length;
  nodes[tile.id] = { boundingVolume: { box: tile.box }, geometricError: tile.geometric_error,
    refine: 'REPLACE', content: { uri } };
  if (tile.parent) (nodes[tile.parent].children ??= []).push(nodes[tile.id]);
}
const rootTile = nodes[plan.tiles[0].id];
assert.equal(plan.tiles[0].parent, null);
// Tileset geometricError is the error of drawing nothing, so use the block's own size:
// the root box diagonal. A smaller value makes Cesium drop the whole block at moderate distance.
const [, , , hx, , , , hy, , , , hz] = plan.tiles[0].box;
const tileset = {
  asset: { version: '1.1', generator }, geometricError: 2 * Math.hypot(hx, hy, hz),
  ...tilesetExtensions,
  root: { transform: plan.root_transform, ...rootTile },
};
const text = JSON.stringify(tileset, null, 2) + '\n';
await writeFile(new URL('tileset.json', root), text);
report.tilesetSha256 = sha(text);
report.planSha256 = sha(await readFile(new URL('tile-plan.json', root)));
report.generator = generator;
await writeFile(new URL('tiles-report.json', root), JSON.stringify(report, null, 2) + '\n');
console.log(`Encoded ${plan.tiles.length} tiles, ${report.totals.splats} splats, ${report.totals.glbBytes} bytes.`);

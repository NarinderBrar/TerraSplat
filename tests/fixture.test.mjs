import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { gunzipSync } from 'node:zlib';

const root = new URL('../public/phase0/', import.meta.url);
const json = name => JSON.parse(readFileSync(new URL(name, root)));
const bytes = readFileSync(new URL('fixture.glb', root));
const gltf = JSON.parse(bytes.subarray(20, 20 + bytes.readUInt32LE(12)).toString());
const binOffset = 28 + bytes.readUInt32LE(12);

test('GLB chunks contain the exact SPZ v2 payload and required decoder profile', () => {
  assert.equal(bytes.readUInt32LE(0), 0x46546c67);
  assert.equal(bytes.readUInt32LE(4), 2);
  assert.equal(bytes.readUInt32LE(8), bytes.length);
  assert.equal(bytes.readUInt32LE(16), 0x4e4f534a);
  assert.equal(bytes.readUInt32LE(binOffset - 4), 0x004e4942);
  assert.equal(bytes.readUInt32LE(binOffset - 8) + binOffset, bytes.length);
  const payload = bytes.subarray(binOffset, binOffset + gltf.bufferViews[0].byteLength);
  assert.deepEqual(payload, readFileSync(new URL('fixture.spz', root)));
  const header = gunzipSync(payload);
  assert.equal(header.readUInt32LE(0), 0x5053474e);
  assert.equal(header.readUInt32LE(4), 2);
  assert.equal(header.readUInt32LE(8), 5);
  assert.equal(header[12], 0); // SH degree
  const primitive = gltf.meshes[0].primitives[0];
  assert.equal(primitive.mode, 0);
  for (const extension of ['KHR_gaussian_splatting', 'KHR_gaussian_splatting_compression_spz_2']) {
    assert.ok(gltf.extensionsRequired.includes(extension));
  }
  assert.equal(primitive.extensions.KHR_gaussian_splatting.extensions.KHR_gaussian_splatting_compression_spz_2.bufferView, 0);
  for (const accessor of gltf.accessors) assert.equal(accessor.count, 5);
});

test('declared bounds cover the fixture and full ellipsoid supports', () => {
  const fixture = json('fixture.json');
  const tileset = json('tileset.json');
  assert.equal(tileset.root.content.uri, 'fixture.glb');
  assert.ok(tileset.geometricError > 0);
  assert.deepEqual(tileset.extensions['3DTILES_content_gltf'].extensionsRequired, gltf.extensionsRequired);
  const accessor = gltf.accessors[0];
  for (const sample of fixture.samples) {
    for (let i = 0; i < 3; i++) {
      assert.ok(sample.gltf[i] >= accessor.min[i] && sample.gltf[i] <= accessor.max[i]);
    }
    assert.ok(Math.hypot(...sample.enu) + 3 * Math.max(...sample.scale) <= tileset.root.boundingVolume.sphere[3]);
  }
});

test('tile transform encodes metric ENU basis at the declared WGS84 anchor', () => {
  const { anchor } = json('fixture.json');
  const m = json('tileset.json').root.transform;
  const lon = anchor.longitude * Math.PI / 180, lat = anchor.latitude * Math.PI / 180;
  const east = [-Math.sin(lon), Math.cos(lon), 0];
  const north = [-Math.sin(lat) * Math.cos(lon), -Math.sin(lat) * Math.sin(lon), Math.cos(lat)];
  const up = [Math.cos(lat) * Math.cos(lon), Math.cos(lat) * Math.sin(lon), Math.sin(lat)];
  for (const [column, expected] of [east, north, up].entries()) {
    expected.forEach((v, i) => assert.ok(Math.abs(m[column * 4 + i] - v) < 1e-7));
  }
  const a = 6378137, e2 = 6.6943799901413165e-3;
  const n = a / Math.sqrt(1 - e2 * Math.sin(lat) ** 2), h = anchor.ellipsoidalHeight;
  const expected = [(n + h) * Math.cos(lat) * Math.cos(lon), (n + h) * Math.cos(lat) * Math.sin(lon), (n * (1 - e2) + h) * Math.sin(lat)];
  expected.forEach((v, i) => assert.ok(Math.abs(m[12 + i] - v) < 1e-6));
});

test('compatibility manifest detects altered fixture assets', () => {
  for (const [name, expected] of Object.entries(json('compatibility.json').hashes)) {
    const actual = createHash('sha256').update(readFileSync(new URL(name, root))).digest('hex');
    assert.equal(actual, expected, name);
  }
});

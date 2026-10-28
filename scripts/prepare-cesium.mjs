import { cp, mkdir } from 'node:fs/promises';
const target = new URL('../public/cesium/', import.meta.url);
await mkdir(target, { recursive: true });
for (const name of ['Cesium.js', 'Assets', 'ThirdParty', 'Widgets', 'Workers']) {
  await cp(new URL(`../node_modules/cesium/Build/Cesium/${name}`, import.meta.url), new URL(name, target), { recursive: true });
}

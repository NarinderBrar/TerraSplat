import { cp, mkdir } from 'node:fs/promises';
await mkdir(new URL('../public/', import.meta.url), { recursive: true });
await cp(new URL('../../../public/cesium/', import.meta.url), new URL('../public/cesium/', import.meta.url), { recursive: true });

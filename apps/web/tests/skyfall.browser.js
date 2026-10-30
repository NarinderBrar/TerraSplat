import { test, expect } from '@playwright/test';
import { existsSync } from 'node:fs';
import { writeFile } from 'node:fs/promises';

test('Skyfall visibility, coarse-to-fine streaming and overlap layers', async ({ page }, info) => {
  test.skip(!existsSync('../../artifacts/releases/skyfall-jax-068-final-r1/release.json'), 'Publish the finished JAX_068 model first');
  const errors = [], glbs = [], external = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('request', r => {
    if (r.url().endsWith('.glb')) glbs.push(r.url());
    if (/^https?:/.test(r.url()) && new URL(r.url()).origin !== 'http://127.0.0.1:5174') external.push(r.url());
  });
  await page.route('**/*', route => {
    const url = route.request().url();
    return /^https?:/.test(url) && new URL(url).origin !== 'http://127.0.0.1:5174' ? route.abort() : route.continue();
  });
  await page.goto('/');
  await expect(page.getByRole('status')).toContainText('Ready');
  expect(glbs).toHaveLength(0);
  await page.getByRole('button', { name: /3D Footprints/ }).click();
  await expect.poll(() => page.evaluate(() => window.terra.viewer.scene.globe.tilesLoaded)).toBe(true);
  await page.screenshot({ path: info.outputPath('skyfall-globe-footprints.png') });
  const camera = async (height, dx = 0, dy = 0, dataset = 'skyfall-jax-068') => page.evaluate(async ({height, dx, dy, dataset}) => {
    const fs = await fetch('/api/footprints').then(r => r.json());
    const f = fs.features.find(f => f.properties.dataset_id === dataset);
    const ring = f.geometry.coordinates[0].slice(0, -1);
    const lon = ring.reduce((s,p) => s+p[0],0)/ring.length;
    const lat = ring.reduce((s,p) => s+p[1],0)/ring.length;
    const C = window.Cesium;
    window.terra.viewer.camera.setView({ destination: C.Cartesian3.fromDegrees(lon+dx, lat+dy, height),
      orientation: { heading: 0, pitch: -Math.PI/2, roll: 0 } });
    window.terra.viewer.scene.requestRender();
  }, {height, dx, dy, dataset});
  await camera(8000);
  await expect.poll(() => glbs.filter(u => u.includes('skyfall-jax-068')).length, { timeout: 120000 }).toBe(1);
  await expect.poll(() => page.evaluate(() => window.terra.stats().visible), { timeout: 120000 }).toBeGreaterThan(0);
  const far = [...glbs];
  expect(far.filter(u => u.includes('jax068-2015'))).toHaveLength(0);
  await camera(220, 0.0015, 0.0013);
  await expect.poll(() => page.evaluate(() => window.terra.stats().visible), { timeout: 120000 }).toBeGreaterThan(1);
  // Deeper trees refine over several passes; tilesReady can be true between them.
  await expect.poll(() => new Set(glbs.filter(u => u.includes('skyfall-jax-068'))).size, { timeout: 120000 }).toBeGreaterThan(1);
  await expect.poll(() => page.evaluate(() => window.terra.tilesReady()), { timeout: 120000 }).toBe(true);
  const fixedRequests = [...new Set(glbs.filter(u => u.includes('skyfall-jax-068')))];
  const total = await page.evaluate(() => fetch('/api/releases/skyfall-jax-068-final-r1').then(r => r.json()).then(m => m.tile_count));
  expect(fixedRequests.length).toBeGreaterThan(1);
  expect(fixedRequests.length).toBeLessThan(total);
  await writeFile(info.outputPath('streaming.json'), JSON.stringify({ far, fixedRequests, total }, null, 2));
  await camera(900);
  await expect.poll(() => page.evaluate(() => window.terra.tilesReady()), { timeout: 120000 }).toBe(true);
  await page.getByRole('button', { name: /3D Footprints/ }).click();
  await page.screenshot({ path: info.outputPath('skyfall-jax.png') });
  await page.getByLabel('Skyfall scenes', { exact: true }).uncheck();
  await expect.poll(() => glbs.some(u => u.includes('jax068-2015-19v-r1'))).toBe(true);
  await expect.poll(() => page.evaluate(() => window.terra.loadedUrls.some(u => u.includes('skyfall-')))).toBe(false);
  await page.getByLabel('Skyfall scenes', { exact: true }).check();
  await page.getByLabel('JAX_068 overlap').selectOption('pipeline');
  await expect.poll(() => page.evaluate(() => window.terra.loadedUrls.some(u => u.includes('skyfall-jax-068')))).toBe(false);
  await expect.poll(() => page.evaluate(() => window.terra.tilesReady()), { timeout: 120000 }).toBe(true);
  await expect(page.getByRole('heading', { name: 'Jacksonville JAX_068 (2015)' })).toBeVisible();
  await page.screenshot({ path: info.outputPath('pipeline-block.png') });
  await page.getByLabel('JAX_068 overlap').selectOption('skyfall');
  await expect.poll(() => page.evaluate(() => window.terra.loadedUrls.some(u => u.includes('jax068-2015')))).toBe(false);
  const nyc = await page.evaluate(() => fetch('/api/datasets').then(r => r.json()).then(ds => ds.find(d => d.dataset_id.startsWith('skyfall-nyc-'))?.dataset_id));
  if (nyc) {
    await camera(1100, 0, 0, nyc);
    await expect.poll(() => glbs.some(u => u.includes(nyc)), { timeout: 120000 }).toBe(true);
    await expect.poll(() => page.evaluate(() => window.terra.tilesReady()), { timeout: 120000 }).toBe(true);
    await page.screenshot({ path: info.outputPath('skyfall-nyc.png') });
  }
  await expect(page.getByRole('alert')).toHaveCount(0);
  expect(errors).toEqual([]); expect(external).toEqual([]);
});

import { test, expect } from '@playwright/test';
import { existsSync } from 'node:fs';
import { writeFile } from 'node:fs/promises';

test('offline globe, footprint navigation, actual splat fragments and release substitution', async ({ page }, info) => {
  test.skip(!existsSync('../../artifacts/releases/catalog.json'), 'Publish local artifacts/releases before running the app browser test');
  const errors = [], external = [], glbs = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('request', request => {
    if (/^https?:/.test(request.url()) && new URL(request.url()).origin !== 'http://127.0.0.1:5174') external.push(request.url());
    if (request.url().endsWith('.glb')) glbs.push(request.url());
  });
  await page.route('**/*', route => {
    const url = route.request().url();
    if (/^https?:/.test(url) && new URL(url).origin !== 'http://127.0.0.1:5174') return route.abort();
    return route.continue();
  });
  await page.goto('/');
  await expect(page.getByRole('status')).toContainText('Ready');
  await expect.poll(() => page.evaluate(() => window.terra?.viewer.imageryLayers.length)).toBe(1);
  expect(await page.evaluate(() => window.terra.stats().height)).toBeGreaterThan(20_000_000);
  // Natural Earth must have reached the canvas, rather than merely being configured.
  await expect.poll(() => page.evaluate(() => window.terra.viewer.scene.globe.tilesLoaded)).toBe(true);
  await page.screenshot({ path: info.outputPath('app-globe.png') });
  await page.getByRole('button', { name: '3D Footprints', exact: false }).click();
  const footprint = page.getByRole('button', { name: /JAX_068 historical 5-view baseline/ });
  await expect(footprint).toBeVisible();
  const datasetCount = await page.evaluate(() => fetch('/api/datasets').then(r => r.json()).then(d => d.length));
  expect(await page.evaluate(() => window.terra.viewer.dataSources.getByName('footprints')[0].entities.values.length)).toBe(datasetCount * 2);
  // Exercise the orbit-scale map marker and its hover/click paths.
  const marker = await page.evaluate(() => {
    const C = window.Cesium, v = window.terra.viewer;
    const entity = v.dataSources.getByName('footprints')[0].entities.values.find(e => e.point);
    const p = C.SceneTransforms.worldToWindowCoordinates(v.scene, entity.position.getValue(v.clock.currentTime));
    return { x: p.x, y: p.y };
  });
  await page.mouse.move(marker.x, marker.y);
  await expect(page.getByRole('tooltip')).toContainText('Registered lidar MAE 2.82 m');
  await page.screenshot({ path: info.outputPath('app-footprints.png') });
  await footprint.click();
  await expect(page.getByRole('heading', { name: 'Jacksonville JAX_068 (2015)' })).toBeVisible();
  await expect.poll(() => page.evaluate(() => window.terra.stats().height), { timeout: 30000 }).toBeLessThan(1500);
  await expect.poll(() => page.evaluate(() => window.terra.stats().visible), { timeout: 120000 }).toBeGreaterThan(1);
  await page.getByRole('button', { name: 'Top-down view' }).click();
  await expect.poll(() => page.evaluate(() => Math.abs(window.terra.viewer.camera.pitch + Math.PI / 2))).toBeLessThan(0.01);
  const location = await page.evaluate(() => {
    const p = window.terra.viewer.camera.positionCartographic, C = window.Cesium;
    return { lon: C.Math.toDegrees(p.longitude), lat: C.Math.toDegrees(p.latitude) };
  });
  expect(location.lat).toBeCloseTo(30.3488, 2); expect(location.lon).toBeCloseTo(-81.6638, 2);
  await expect.poll(() => page.evaluate(() => window.terra.stats().visible), { timeout: 120000 }).toBeGreaterThan(0);
  await expect.poll(() => page.evaluate(() => window.terra.tilesReady()), { timeout: 120000 }).toBe(true);
  await page.getByRole('button', { name: '3D Footprints', exact: false }).click();
  // Compare the same frame with and without splats: base imagery is coloured too.
  const measure = () => page.evaluate(async () => {
    const map = window.terra, v = map.viewer;
    const manifest = await fetch('/api/releases/jax068-2015-5v-r2').then(r => r.json());
    const corners = map.projectFootprint(manifest);
    const canvas = v.canvas, gl = canvas.getContext('webgl2');
    const read = () => {
      v.render(); const pixels = new Uint8Array(canvas.width * canvas.height * 4);
      gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels); return pixels;
    };
    const before = read(); v.scene.primitives.show = false;
    const without = read(); v.scene.primitives.show = true; v.scene.requestRender();
    const inside = (x, y) => {
      let hit = false;
      for (let i = 0, j = corners.length - 1; i < corners.length; j = i++) {
        const a = corners[i], b = corners[j];
        if ((a.y > y) !== (b.y > y) && x < (b.x - a.x) * (y - a.y) / (b.y - a.y) + a.x) hit = !hit;
      } return hit;
    };
    let changed = 0, changedInside = 0, footprintPixels = 0;
    for (let y = 0; y < canvas.height; y++) for (let x = 0; x < canvas.width; x++) {
      const i = ((canvas.height - 1 - y) * canvas.width + x) * 4;
      const diff = Math.abs(before[i] - without[i]) + Math.abs(before[i+1] - without[i+1]) + Math.abs(before[i+2] - without[i+2]) > 24;
      const within = inside(x, y); changed += diff; changedInside += diff && within; footprintPixels += within;
    }
    return { corners, changed, insideFraction: changedInside / changed, coverage: changedInside / footprintPixels };
  });
  await expect.poll(async () => (await measure()).changed, { timeout: 120000 }).toBeGreaterThan(10000);
  // Splats draw per tile once its worker sort finishes, and hiding primitives to measure triggers a re-sort, so
  // consecutive measurements can differ. Wait for the block to fill and assert on that same measurement.
  let pixels;
  await expect.poll(async () => (pixels = await measure()).coverage, { timeout: 120000 }).toBeGreaterThan(0.6);
  expect(pixels.insideFraction).toBeGreaterThan(0.85);
  expect(pixels.coverage).toBeGreaterThan(0.6);
  await writeFile(info.outputPath('app-pixels.json'), JSON.stringify(pixels, null, 2));
  await page.screenshot({ path: info.outputPath('app-splats-top.png') });
  await page.evaluate(async () => {
    const features = await fetch('/api/footprints').then(r => r.json());
    await window.terra.flyTo(features.features[0]);
  });
  await expect.poll(async () => (await measure()).changed, { timeout: 120000 }).toBeGreaterThan(10000);
  await page.screenshot({ path: info.outputPath('app-splats-oblique.png') });
  await page.getByText('Limitations & evidence', { exact: true }).click();
  await expect(page.getByText(/No ground control; independent geographic accuracy/)).toBeVisible();
  await expect(page.getByRole('link', { name: 'result-2015-8w.json' })).toHaveAttribute('href', /releases\/jax068-2015-5v-r2\/evidence/);
  await page.getByText('Limitations & evidence', { exact: true }).click();
  await page.getByLabel('Release · historical 2015').selectOption('jax068-2015-3v-r2');
  await expect(page.getByRole('status')).toContainText('Ready');
  await expect.poll(() => page.evaluate(() => window.terra.loadedUrls.join(','))).toContain('jax068-2015-3v-r2/tileset.json');
  await expect.poll(() => glbs.some(url => url.includes('jax068-2015-3v-r2'))).toBe(true);
  await expect.poll(() => page.evaluate(() => window.terra.stats().visible), { timeout: 120000 }).toBeGreaterThan(0);
  await page.getByLabel('Tile diagnostics').check();
  await expect(page.getByTestId('tile-stats')).toContainText('visible');
  await page.screenshot({ path: info.outputPath('app-release-3v.png') });
  await page.getByLabel('Release · historical 2015').selectOption('jax068-2015-5v-r2');
  await expect.poll(() => page.evaluate(() => window.terra.loadedUrls.join(','))).toContain('jax068-2015-5v-r2/tileset.json');
  // Keyboard access and narrow layout remain usable.
  await page.getByRole('button', { name: '3D Footprints', exact: false }).click();
  await footprint.focus(); await page.keyboard.press('Enter');
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByLabel('Tile diagnostics').uncheck();
  await page.screenshot({ path: info.outputPath('app-narrow.png') });
  await page.getByRole('button', { name: 'Collapse panel' }).click();
  await expect(page.getByRole('button', { name: 'Expand panel' })).toBeVisible();
  await expect(page.getByRole('alert')).toHaveCount(0);
  expect(errors).toEqual([]); expect(external).toEqual([]);
});

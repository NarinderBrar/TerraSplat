import { test, expect } from '@playwright/test';
import { existsSync } from 'node:fs';
import { writeFile } from 'node:fs/promises';

const tileset = new URL('../public/tiles/pilot-2015-04/tileset.json', import.meta.url);

test('2015 pilot tiles render inside the block footprint and refine by distance', async ({ page }, testInfo) => {
  test.skip(!existsSync(tileset), 'Pilot tiles are generated locally into artifacts/tiles');
  test.setTimeout(240000);
  const errors = [];
  const external = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') console.log('BROWSER', message.text()); });
  page.on('request', request => { if (/^https?:/.test(request.url()) && new URL(request.url()).origin !== 'http://127.0.0.1:4173') external.push(request.url()); });
  await page.goto('/pilot.html');
  await expect(page.getByRole('status')).toContainText('Pilot tiles loaded', { timeout: 120000 });

  async function settle(view) {
    await page.evaluate(view => {
      const { views, viewer, frame } = window.pilot;
      if (typeof view === 'string') views[view]();
      else viewer.camera.lookAtTransform(frame, new window.Cesium.Cartesian3(...view));
    }, view);
    await expect.poll(() => page.evaluate(() => { window.pilot.viewer.render(); return window.pilot.tileset.tilesLoaded; }),
      { timeout: 120000 }).toBe(true);
    return page.evaluate(() => window.pilot.visibleTiles());
  }

  // Top view: rendered splats should fall inside the projected 250 m block square.
  await settle('top');
  const measure = () => page.evaluate(() => {
    const { viewer, project, ground, plan } = window.pilot;
    viewer.render();
    const half = plan.crop.half_extent_m;
    const corners = [[-half, -half], [half, -half], [half, half], [-half, half]].map(([e, n]) => project([e, n, ground]));
    const canvas = viewer.canvas;
    const gl = canvas.getContext('webgl2');
    const pixels = new Uint8Array(canvas.width * canvas.height * 4);
    gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    const rect = canvas.getBoundingClientRect();
    const inside = (x, y) => {
      let hit = false;
      for (let i = 0, j = corners.length - 1; i < corners.length; j = i++) {
        const a = corners[i], b = corners[j];
        if ((a.y > y) !== (b.y > y) && x < (b.x - a.x) * (y - a.y) / (b.y - a.y) + a.x) hit = !hit;
      }
      return hit;
    };
    let coloured = 0, colouredInside = 0, footprintPixels = 0, footprintColoured = 0;
    for (let y = 0; y < canvas.height; y++) for (let x = 0; x < canvas.width; x++) {
      const o = ((canvas.height - 1 - y) * canvas.width + x) * 4;
      const lit = Math.abs(pixels[o] - 11) + Math.abs(pixels[o + 1] - 15) + Math.abs(pixels[o + 2] - 20) > 24;
      const within = inside(x + rect.left, y + rect.top);
      coloured += lit; colouredInside += lit && within;
      footprintPixels += within; footprintColoured += lit && within;
    }
    return { corners, coloured, insideFraction: colouredInside / coloured, footprintCoverage: footprintColoured / footprintPixels };
  });
  // Sorting runs on workers; wait for actual splat fragments.
  await expect.poll(async () => (await measure()).coloured, { timeout: 120000 }).toBeGreaterThan(10000);
  const footprint = await measure();
  await page.screenshot({ path: testInfo.outputPath('pilot-top.png') });
  await writeFile(testInfo.outputPath('footprint.json'), JSON.stringify(footprint, null, 2) + '\n');
  expect(footprint.coloured).toBeGreaterThan(10000);
  expect(footprint.insideFraction, 'rendered pixels inside the block square').toBeGreaterThan(0.85);
  expect(footprint.footprintCoverage, 'block square covered by splats').toBeGreaterThan(0.85);

  // REPLACE refinement: far away only the merged root draws; close up, full-resolution leaves draw.
  expect(await settle([0, -1, 8000])).toEqual(['r.glb']);
  const near = await settle([60, 60, 120]);
  expect(near.some(name => /^r\d\d\.glb$/.test(name)), `near tiles ${near}`).toBe(true);
  expect(near).not.toContain('r.glb');

  await settle('oblique');
  await page.screenshot({ path: testInfo.outputPath('pilot-oblique.png') });
  expect(errors).toEqual([]);
  expect(external).toEqual([]);
});

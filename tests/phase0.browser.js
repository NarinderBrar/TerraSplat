import { test, expect } from '@playwright/test';

test('SPZ fixture renders at the expected metric positions and orientations', async ({ page }, testInfo) => {
  const errors = [];
  const external = [];
  page.on('pageerror', error => { errors.push(error.message); console.log('PAGE ERROR', error.message); });
  page.on('console', message => { if (message.type() === 'error') console.log('BROWSER', message.text()); });
  page.on('requestfailed', request => console.log('REQUEST FAILED', request.url(), request.failure()));
  page.on('request', request => { if (/^https?:/.test(request.url()) && new URL(request.url()).origin !== 'http://127.0.0.1:4173') external.push(request.url()); });
  await page.goto('/');
  await expect(page.getByRole('status')).toContainText('Fixture loaded', { timeout: 20000 });
  async function measure() {
    return page.evaluate(() => {
      const { viewer, fixture, project } = window.phase0;
      viewer.render();
      const canvas = viewer.canvas;
      const gl = canvas.getContext('webgl2');
      const pixels = new Uint8Array(canvas.width * canvas.height * 4);
      gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
      const rect = canvas.getBoundingClientRect();
      const measurements = {};
      for (const sample of fixture.samples.filter(s => s.name !== 'origin')) {
        const center = project(sample.enu);
        const coords = [];
        for (let y = 0; y < canvas.height; y++) for (let x = 0; x < canvas.width; x++) {
          if (Math.hypot(x + rect.left - center.x, y + rect.top - center.y) > 65) continue;
          const offset = ((canvas.height - 1 - y) * canvas.width + x) * 4;
          const [r, g, b] = pixels.subarray(offset, offset + 3);
          const matches = sample.name === 'east' ? r > 60 && r > g * 2 && r > b * 2
            : sample.name === 'north' ? g > 60 && g > r * 2 && g > b * 2
            : sample.name === 'up' ? b > 60 && b > r * 2 && b > g * 2
            : r > 45 && g > 25 && r > g * 1.2 && g > b * 2;
          if (matches) coords.push({ x: x + rect.left, y: y + rect.top, weight: Math.max(r, g, b) });
        }
        const weight = coords.reduce((sum, p) => sum + p.weight, 0);
        const mx = coords.reduce((sum, p) => sum + p.x * p.weight, 0) / weight;
        const my = coords.reduce((sum, p) => sum + p.y * p.weight, 0) / weight;
        const xx = coords.reduce((sum, p) => sum + (p.x - mx) ** 2 * p.weight, 0) / weight;
        const yy = coords.reduce((sum, p) => sum + (p.y - my) ** 2 * p.weight, 0) / weight;
        const xy = coords.reduce((sum, p) => sum + (p.x - mx) * (p.y - my) * p.weight, 0) / weight;
        const majorAxis = 0.5 * Math.atan2(2 * xy, xx - yy);
        const expectedDirection = sample.name === 'east' ? [1, 0, 0] : sample.name === 'north' ? [0, 1, 0]
          : sample.name === 'up' ? [0, 0, 1] : [Math.SQRT1_2, 0, Math.SQRT1_2];
        const tip = project(sample.enu.map((v, i) => v + expectedDirection[i]));
        const expectedAngle = Math.atan2(tip.y - center.y, tip.x - center.x);
        const angleError = Math.acos(Math.min(1, Math.abs(Math.cos(majorAxis - expectedAngle))));
        measurements[sample.name] = { count: coords.length, centroidError: Math.hypot(mx - center.x, my - center.y),
          majorAxis, expectedAngle, angleError, peak: Math.max(0, ...coords.map(p => p.weight)),
          xx, yy, xy, center };
      }
      return measurements;
    });
  }
  // Loading and sorting run on workers; wait for actual colored fragments.
  await expect.poll(async () => (await measure()).east.count).toBeGreaterThan(25);
  const measurements = await measure();
  for (const [name, result] of Object.entries(measurements)) {
    expect(result.count, `${name} rendered fragments`).toBeGreaterThan(25);
    expect(result.centroidError, `${name} placement in pixels`).toBeLessThan(3);
    expect(result.angleError, `${name} ellipse orientation radians`).toBeLessThan(0.15);
  }
  expect(measurements.east.peak).toBeGreaterThan(170);
  expect(measurements.rotated.peak).toBeLessThan(measurements.east.peak * 0.8);
  const distances = await page.evaluate(() => {
    const { world } = window.phase0;
    return [[10, 0, 0], [0, 10, 0], [0, 0, 10]].map(p => window.Cesium.Cartesian3.distance(world([0, 0, 0]), world(p)));
  });
  for (const distance of distances) expect(distance).toBeCloseTo(10, 6);
  await page.screenshot({ path: testInfo.outputPath('fixture.png') });
  await testInfo.attach('render-measurements', { body: JSON.stringify(measurements, null, 2), contentType: 'application/json' });
  await page.getByLabel('Show ellipsoid').check();
  await page.getByLabel('Show tile bounds').check();
  await page.getByRole('button', { name: 'Reset camera' }).click();
  await expect.poll(() => page.evaluate(() => window.phase0.viewer.scene.globe.show && window.phase0.tileset.debugShowBoundingVolume)).toBe(true);
  expect(errors).toEqual([]);
  expect(external).toEqual([]);
});

import { defineConfig } from '@playwright/test';
import { existsSync } from 'node:fs';
const releases = existsSync('../../artifacts/releases/catalog.json');
export default defineConfig({
  testDir: './tests', testMatch: '**/*.browser.js', workers: 1, timeout: 240000,
  use: {
    baseURL: 'http://127.0.0.1:5174', viewport: { width: 1280, height: 900 },
    launchOptions: { executablePath: process.env.CHROME_PATH || '/usr/bin/google-chrome',
      args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] },
    screenshot: 'only-on-failure', trace: 'retain-on-failure',
  },
  webServer: releases ? [
    { command: 'cd ../.. && .venv-api/bin/python -m uvicorn services.api.main:app --host 127.0.0.1 --port 8001', url: 'http://127.0.0.1:8001/api/health', reuseExistingServer: false },
    { command: 'npm run prepare:cesium && npm exec vite -- --port 5174', url: 'http://127.0.0.1:5174', reuseExistingServer: false,
      env: { VITE_REMOTE_IMAGERY: 'false', VITE_TEST_BRIDGE: 'true', TERRASPLAT_API_PROXY: 'http://127.0.0.1:8001' } },
  ] : [],
});

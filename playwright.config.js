import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './tests', testMatch: '**/*.browser.js', workers: 1, timeout: 60000,
  use: {
    baseURL: 'http://127.0.0.1:4173', viewport: { width: 1280, height: 900 },
    launchOptions: { executablePath: process.env.CHROME_PATH || '/usr/bin/google-chrome',
      args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] },
    screenshot: 'only-on-failure',
  },
  webServer: { command: 'npm run preview -- --host 127.0.0.1 --port 4173 --strictPort',
    url: 'http://127.0.0.1:4173', reuseExistingServer: false },
});

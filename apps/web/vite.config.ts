import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
// Defaults to loopback; TERRASPLAT_WEB_HOST=0.0.0.0 (npm run dev:app:lan) exposes it to the local network.
const host = process.env.TERRASPLAT_WEB_HOST || '127.0.0.1';
const proxy = Object.fromEntries(['/api', '/releases'].map(path => [path, {
  target: process.env.TERRASPLAT_API_PROXY || 'http://127.0.0.1:8000', changeOrigin: true,
}]));
export default defineConfig({
  plugins: [react()],
  server: { host, port: 5173, strictPort: true, proxy },
  preview: { host, proxy },
  test: { include: ['tests/**/*.test.ts'] },
});

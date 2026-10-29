import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
const proxy = Object.fromEntries(['/api', '/releases'].map(path => [path, {
  target: process.env.TERRASPLAT_API_PROXY || 'http://127.0.0.1:8000', changeOrigin: true,
}]));
export default defineConfig({
  plugins: [react()],
  server: { host: '127.0.0.1', port: 5173, strictPort: true, proxy },
  preview: { host: '127.0.0.1', proxy },
  test: { include: ['tests/**/*.test.ts'] },
});

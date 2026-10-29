import { defineConfig } from 'vite';

export default defineConfig({
  build: { rollupOptions: { input: { main: 'index.html', pilot: 'pilot.html' } } },
  // The repo root holds caches, venvs, toolchains and run outputs; watching them exhausts inotify.
  server: {
    watch: {
      ignored: ['**/.cache/**', '**/.venv*/**', '**/.toolchains/**', '**/.external/**', '**/artifacts/**',
        '**/apps/**', '**/Ref/**', '**/TerraScope/**', '**/data/**', '**/checkpoints/**'],
    },
  },
});

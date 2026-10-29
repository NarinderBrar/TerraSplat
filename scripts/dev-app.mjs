import { spawn } from 'node:child_process';
const children = [
  spawn('.venv-api/bin/python', ['-m', 'uvicorn', 'services.api.main:app', '--host', '127.0.0.1', '--port', '8000'], { stdio: 'inherit' }),
  spawn('npm', ['--prefix', 'apps/web', 'run', 'dev'], { stdio: 'inherit' }),
];
let stopping = false;
function stop(code = 0) {
  if (stopping) return;
  stopping = true;
  children.forEach(child => child.kill('SIGTERM'));
  process.exitCode = code;
}
for (const child of children) {
  child.on('error', error => { console.error(error); stop(1); });
  child.on('exit', code => stop(code ?? 1));
}
process.on('SIGINT', () => stop());
process.on('SIGTERM', () => stop());

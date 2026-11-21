// Copy committed website assets into Vite's generated public directory.
import { cp, mkdir } from 'node:fs/promises';
const root = new URL('../', import.meta.url);
for (const name of ['sample', 'marketing']) {
  const destination = new URL(`web-resources/${name}/`, root);
  await mkdir(destination, { recursive: true });
  await cp(new URL(`${name}/`, root), destination, { recursive: true });
}

import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// Respect an externally-assigned port (e.g. the dev preview harness sets
// PORT when 5173 is already taken by another project) instead of always
// binding to Vite's hardcoded default.
const port = Number(process.env.PORT) || 5173;

export default defineConfig({
  plugins: [svelte()],
  build: {
    rollupOptions: {
      input: {
        landing: fileURLToPath(new URL("./index.html", import.meta.url)),
        demo: fileURLToPath(new URL("./demo/index.html", import.meta.url)),
      },
    },
  },
  // vendored external assets (Verovio now, the MuseScore soundfont later) are
  // synced into web-resources/ via `npm run sync-verovio` rather than
  // committed twice; see web/scripts/sync-verovio.mjs.
  publicDir: "web-resources",
  server: {
    port,
    // Falls back to the next free port instead of hard-failing when 5173 is
    // taken - a real, permanent condition on at least one dev machine this
    // runs on (another project holds 5173). The dev preview harness works
    // around this itself by setting PORT before launch; strictPort:false
    // makes a plain `npm run dev`/`dev:all` in a regular terminal resilient
    // to the same collision without that harness.
    strictPort: false,
  },
});

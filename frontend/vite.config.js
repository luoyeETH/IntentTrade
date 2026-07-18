import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const here = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [react()],
  base: "/static/",
  build: {
    outDir: path.resolve(here, "../src/intent_trade/web/static"),
    emptyOutDir: true,
    // Hash CSS + JS so browsers / CDN (e.g. Cloudflare on xtrade.luoye.de)
    // never keep a stale style.css after deploy.
    rollupOptions: {
      output: {
        assetFileNames: "assets/[name]-[hash][extname]",
        chunkFileNames: "assets/[name]-[hash].js",
        entryFileNames: "assets/[name]-[hash].js",
      },
    },
  },
});

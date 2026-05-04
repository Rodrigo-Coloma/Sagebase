import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Sub-path support: build with VITE_BASE_PATH=/medlit/ to deploy under /medlit.
// Defaults to "/" (root). Affects asset URLs in index.html and import.meta.env.BASE_URL,
// which we feed into React Router's basename in main.tsx.
export default defineConfig(() => ({
  base: process.env.VITE_BASE_PATH || "/",
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
}));

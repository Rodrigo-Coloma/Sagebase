import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// API base is read from env at build time.
//   - In dev: defaults to http://localhost:8000 with a vite proxy
//   - In prod: nginx serves /api/* to the FastAPI service
export default defineConfig({
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
});

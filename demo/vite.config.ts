import { defineConfig } from "vite";

// onnxruntime-web ships wasm assets; exclude from pre-bundling so the worker
// and wasm files resolve correctly. MediaPipe loads its wasm from a CDN/local.
export default defineConfig({
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  server: { port: 5173 },
});

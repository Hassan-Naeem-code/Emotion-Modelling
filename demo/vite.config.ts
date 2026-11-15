import { defineConfig } from "vite";

// onnxruntime-web ships wasm assets; exclude from pre-bundling so the worker
// and wasm files resolve correctly. MediaPipe loads its wasm from a CDN/local.
// Served at hassan-naeem.com/emotion-modeling (the portfolio rewrites that path here); vercel.json maps the
// prefix back to the root so the plain va-uncertainty-demo.vercel.app address keeps working too.
export default defineConfig({
  base: "/emotion-modeling/",
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  server: { port: 5173 },
});

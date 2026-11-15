import { resolve } from "node:path";
import { readFileSync } from "node:fs";
import { defineConfig, type Connect, type Plugin } from "vite";

/**
 * Make the local dev and preview servers route like vercel.json does in
 * production: /app -> /app/, and /results(/) -> the static results page in
 * public/. Without this, both fall back to the landing page locally.
 */
function pageRoutes(): Plugin {
  const routes: Connect.NextHandleFunction = (req, res, next) => {
    const url = (req.url ?? "").split("?")[0];
    if (url === "/emotion-modeling/app") {
      res.statusCode = 301;
      res.setHeader("Location", "/emotion-modeling/app/");
      return res.end();
    }
    if (url === "/emotion-modeling/results" || url === "/emotion-modeling/results/") {
      res.setHeader("Content-Type", "text/html; charset=utf-8");
      return res.end(readFileSync(resolve(__dirname, "public/results/index.html")));
    }
    next();
  };
  return {
    name: "page-routes",
    configureServer: (server) => void server.middlewares.use(routes),
    configurePreviewServer: (server) => void server.middlewares.use(routes),
  };
}

// onnxruntime-web ships wasm assets; exclude from pre-bundling so the worker
// and wasm files resolve correctly. MediaPipe loads its wasm from a CDN/local.
// Served at hassan-naeem.com/emotion-modeling (the portfolio rewrites that path here); vercel.json maps the
// prefix back to the root so the plain va-uncertainty-demo.vercel.app address keeps working too.
export default defineConfig({
  base: "/emotion-modeling/",
  plugins: [pageRoutes()],
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  server: { port: 5173 },
  // Two pages: the landing page at the root and the live app at /app/.
  build: {
    rollupOptions: {
      input: { main: resolve(__dirname, "index.html"), app: resolve(__dirname, "app/index.html") },
    },
  },
});

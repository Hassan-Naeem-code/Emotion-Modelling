// Copies the WebAssembly runtimes the demo needs from node_modules into public/vendor, so the page loads them
// from its own origin (no third-party CDN, so a strict Content-Security-Policy can allow only 'self').
// Runs before `vite` and `vite build`; public/vendor is generated and git-ignored.
import { cpSync, mkdirSync, rmSync } from "node:fs";

const out = new URL("../public/vendor/", import.meta.url);
rmSync(out, { recursive: true, force: true });
const copy = (from, files, to) => {
  mkdirSync(new URL(to, out), { recursive: true });
  for (const f of files) cpSync(new URL(`../node_modules/${from}${f}`, import.meta.url), new URL(`${to}${f}`, out));
};
// onnxruntime-web 1.18 wasm backend. The bundle names four files; with numThreads = 1 only the single-threaded
// pair is ever requested (SIMD build, plain build as the fallback for browsers without SIMD).
copy("onnxruntime-web/dist/", ["ort-wasm-simd.wasm", "ort-wasm.wasm"], "ort/");
// MediaPipe tasks-vision 0.10.14 face landmarker runtime
copy("@mediapipe/tasks-vision/wasm/", [
  "vision_wasm_internal.js", "vision_wasm_internal.wasm", "vision_wasm_nosimd_internal.js", "vision_wasm_nosimd_internal.wasm",
], "mediapipe/");


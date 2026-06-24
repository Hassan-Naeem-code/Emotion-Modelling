/**
 * Model inference via ONNX Runtime Web (on-device). Loads model.onnx +
 * model_meta.json from /public if present. If no model is bundled (fresh
 * checkout), falls back to a transparent SYNTHETIC predictor so the UI — and
 * crucially the uncertainty/abstention UX — works end-to-end at v0.
 *
 * Contract from export/to_onnx.py: outputs va_mean (1,2) and va_std (1,2).
 */
import * as ort from "onnxruntime-web";
import { THRESHOLDS } from "./config";

// ONNX Runtime Web needs its WebAssembly runtime files. Without this, session
// creation throws and the whole demo silently falls back to the placeholder.
// Load the wasm from a CDN and run single-threaded so we don't require the
// SharedArrayBuffer / cross-origin-isolation headers a dev server lacks.
ort.env.wasm.wasmPaths =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";
ort.env.wasm.numThreads = 1;

export interface ModelMeta {
  image_size: number;
  norm_mean: [number, number, number];
  norm_std: [number, number, number];
  head: string;
  abstain_std_threshold: number;
}

export interface Prediction {
  valence: number;
  arousal: number;
  std: number; // scalar predictive std (mean over dims)
}

const DEFAULT_META: ModelMeta = {
  image_size: 224,
  norm_mean: [0.485, 0.456, 0.406],
  norm_std: [0.229, 0.224, 0.225],
  head: "synthetic",
  abstain_std_threshold: THRESHOLDS.uncertaintyStd,
};

export class AffectPredictor {
  private session: ort.InferenceSession | null = null;
  meta: ModelMeta = DEFAULT_META;
  isReal = false;

  async load(): Promise<void> {
    try {
      const metaResp = await fetch("/model_meta.json");
      if (metaResp.ok) this.meta = await metaResp.json();
      this.session = await ort.InferenceSession.create("/model.onnx", {
        executionProviders: ["wasm"],
      });
      this.isReal = true;
    } catch (e) {
      // No model bundled OR the runtime failed to init — run in synthetic mode.
      console.warn("[AffectPredictor] model load failed, using synthetic:", e);
      this.session = null;
      this.isReal = false;
      this.meta = DEFAULT_META;
    }
  }

  get label(): string {
    return this.isReal
      ? `ONNX (${this.meta.head})`
      : "SYNTHETIC placeholder (no trained model bundled)";
  }

  /** Preprocess a face-crop canvas into a normalized NCHW Float32 tensor. */
  private preprocess(crop: HTMLCanvasElement): ort.Tensor {
    const s = this.meta.image_size;
    const ctx = crop.getContext("2d", { willReadFrequently: true })!;
    const { data } = ctx.getImageData(0, 0, s, s);
    const out = new Float32Array(3 * s * s);
    const [mr, mg, mb] = this.meta.norm_mean;
    const [sr, sg, sb] = this.meta.norm_std;
    const plane = s * s;
    for (let i = 0, px = 0; i < data.length; i += 4, px++) {
      out[px] = (data[i] / 255 - mr) / sr;
      out[plane + px] = (data[i + 1] / 255 - mg) / sg;
      out[2 * plane + px] = (data[i + 2] / 255 - mb) / sb;
    }
    return new ort.Tensor("float32", out, [1, 3, s, s]);
  }

  async predict(crop: HTMLCanvasElement, faceX: number, faceY: number): Promise<Prediction> {
    if (!this.session) return this.synthetic(faceX, faceY);
    const input = this.preprocess(crop);
    const res = await this.session.run({ input });
    const mean = res.va_mean.data as Float32Array;
    const std = res.va_std.data as Float32Array;
    return {
      valence: clamp(mean[0]),
      arousal: clamp(mean[1]),
      std: (std[0] + std[1]) / 2,
    };
  }

  /**
   * Synthetic predictor: maps face position to a plausible VA point with
   * position-dependent uncertainty. Deliberately NOT confident — it exists only
   * so the abstention/visualization pipeline is exercisable without a model.
   */
  private synthetic(faceX: number, faceY: number): Prediction {
    const valence = clamp(Math.sin(faceX * Math.PI * 2) * 0.5);
    const arousal = clamp(Math.cos(faceY * Math.PI * 2) * 0.5);
    // Uncertainty grows toward frame edges (proxy for unreliable framing).
    const edge = Math.max(
      Math.abs(faceX - 0.5),
      Math.abs(faceY - 0.5),
    );
    const std = 0.15 + edge * 0.6;
    return { valence, arousal, std };
  }
}

function clamp(v: number): number {
  return Math.max(-1, Math.min(1, v));
}

export interface EmotionResult {
  label: string;
  confidence: number; // 0..1 (top-class probability)
  probs: { label: string; p: number }[]; // sorted desc
}

/**
 * Categorical emotion classifier (neutral/happy/sad/surprise/fear/disgust/anger).
 * Loaded from /emotion.onnx if present; if absent the demo simply hides the
 * emotion readout (graceful). This gives a recognizable, responsive label
 * alongside the valence/arousal + uncertainty view.
 */
export class EmotionPredictor {
  private session: ort.InferenceSession | null = null;
  private classes: string[] = [];
  private size = 224;
  private mean: number[] = [0.485, 0.456, 0.406];
  private std: number[] = [0.229, 0.224, 0.225];
  available = false;
  valAcc: number | null = null;

  async load(): Promise<void> {
    try {
      const meta = await (await fetch("/emotion_meta.json")).json();
      this.classes = meta.classes;
      this.size = meta.image_size;
      this.mean = meta.norm_mean;
      this.std = meta.norm_std;
      this.valAcc = meta.val_acc ?? null;
      this.session = await ort.InferenceSession.create("/emotion.onnx", {
        executionProviders: ["wasm"],
      });
      this.available = true;
    } catch (e) {
      console.warn("[EmotionPredictor] model load failed:", e);
      this.available = false;
    }
  }

  async predict(crop: HTMLCanvasElement): Promise<EmotionResult | null> {
    if (!this.session) return null;
    const s = this.size;
    const ctx = crop.getContext("2d", { willReadFrequently: true })!;
    const { data } = ctx.getImageData(0, 0, s, s);
    const out = new Float32Array(3 * s * s);
    const plane = s * s;
    for (let i = 0, px = 0; i < data.length; i += 4, px++) {
      out[px] = (data[i] / 255 - this.mean[0]) / this.std[0];
      out[plane + px] = (data[i + 1] / 255 - this.mean[1]) / this.std[1];
      out[2 * plane + px] = (data[i + 2] / 255 - this.mean[2]) / this.std[2];
    }
    const res = await this.session.run({
      input: new ort.Tensor("float32", out, [1, 3, s, s]),
    });
    const probs = Array.from(res.probs.data as Float32Array);
    const ranked = probs
      .map((p, i) => ({ label: this.classes[i], p }))
      .sort((a, b) => b.p - a.p);
    return { label: ranked[0].label, confidence: ranked[0].p, probs: ranked };
  }
}

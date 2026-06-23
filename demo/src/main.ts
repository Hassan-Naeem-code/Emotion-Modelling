/**
 * App entry: consent gate -> camera -> per-frame loop:
 *   webcam -> MediaPipe landmarks -> aligned crop -> ONNX model
 *   -> abstention decision -> VA plane + details panel.
 *
 * Everything runs on-device. The uncertainty/abstention UX is real from v0 even
 * with the synthetic placeholder model.
 */
import { decide, type Decision } from "./abstain";
import { frameBrightness, startCamera, stopCamera } from "./camera";
import { PRIVACY, THRESHOLDS } from "./config";
import {
  createLandmarker,
  detect,
  landmarkMotion,
  type FaceInfo,
} from "./landmarker";
import { AffectPredictor, type Prediction } from "./model";
import { coarseLabel } from "./readout";
import { VAPlane } from "./viz";
import type { FaceLandmarker } from "@mediapipe/tasks-vision";

const $ = (id: string) => document.getElementById(id)!;

// --- populate auditable copy from config ---
$("consent-copy").textContent = PRIVACY.consentBody;
($("fineprint") as HTMLElement).textContent = PRIVACY.fineprint;
($("live-fineprint") as HTMLElement).textContent = PRIVACY.liveFineprint;
const ul = $("privacy-points");
for (const p of PRIVACY.points) {
  const li = document.createElement("li");
  li.textContent = p;
  ul.appendChild(li);
}

const video = $("video") as HTMLVideoElement;
const plane = new VAPlane($("va-plane") as HTMLCanvasElement);
const scratch = document.createElement("canvas");
const cropCanvas = document.createElement("canvas");

let stream: MediaStream | null = null;
let landmarker: FaceLandmarker | null = null;
const predictor = new AffectPredictor();

let prevLandmarks: { x: number; y: number }[] | null = null;
let noFaceStreak = 0;
let smoothed: Prediction | null = null;
let running = false;

/** Build an aligned square crop of the face from the bbox for the model. */
function cropFace(face: FaceInfo): HTMLCanvasElement {
  const s = predictor.meta.image_size;
  cropCanvas.width = s;
  cropCanvas.height = s;
  const ctx = cropCanvas.getContext("2d", { willReadFrequently: true })!;
  const vw = video.videoWidth || 640;
  const vh = video.videoHeight || 480;
  if (face.bbox) {
    // Build a SQUARE crop centred on the face (matches AffectNet's square face
    // crops). Stretching a tall bbox into a square distorts the face and skews
    // the VA estimate, so we take the larger side + padding and clamp to frame.
    const [bx, by, bw, bh] = face.bbox;
    const pad = 0.25;
    const cx = (bx + bw / 2) * vw;
    const cy = (by + bh / 2) * vh;
    let side = Math.max(bw * vw, bh * vh) * (1 + 2 * pad);
    side = Math.min(side, vw, vh);
    let x = cx - side / 2;
    let y = cy - side / 2;
    x = Math.max(0, Math.min(x, vw - side));
    y = Math.max(0, Math.min(y, vh - side));
    ctx.drawImage(video, x, y, side, side, 0, 0, s, s);
  } else {
    ctx.drawImage(video, 0, 0, s, s);
  }
  return cropCanvas;
}

function updateDetails(pred: Prediction | null, d: Decision) {
  $("d-valence").textContent = pred ? pred.valence.toFixed(2) : "—";
  $("d-arousal").textContent = pred ? pred.arousal.toFixed(2) : "—";
  // Derived label only when we're actually committing to an estimate.
  $("d-readout").textContent =
    pred && !d.abstain ? coarseLabel(pred.valence, pred.arousal) : "—";
  $("d-confidence").textContent = `${Math.round(d.confidence * 100)}%`;
  $("d-quality").textContent = `${Math.round(d.quality * 100)}%`;
  $("d-reason").textContent = d.reason ?? "—";
  $("d-model").textContent = predictor.label;
  const badge = $("state-badge");
  if (d.abstain) {
    badge.textContent = `NOT SURE — ${d.reason}`;
    badge.className = "state abstain";
  } else {
    badge.textContent = "estimate";
    badge.className = "state ok";
  }
}

function smooth(pred: Prediction): Prediction {
  if (!smoothed) {
    smoothed = pred;
    return pred;
  }
  const a = THRESHOLDS.smoothing;
  smoothed = {
    valence: a * smoothed.valence + (1 - a) * pred.valence,
    arousal: a * smoothed.arousal + (1 - a) * pred.arousal,
    std: a * smoothed.std + (1 - a) * pred.std,
  };
  return smoothed;
}

async function loop() {
  if (!running || !landmarker) return;
  const tsMs = performance.now();
  const face = detect(landmarker, video, tsMs);
  const brightness = frameBrightness(video, scratch);
  const motion = landmarkMotion(prevLandmarks, face.landmarks);
  prevLandmarks = face.landmarks;
  noFaceStreak = face.present ? 0 : noFaceStreak + 1;

  let pred: Prediction | null = null;
  if (face.present && face.bbox) {
    const cx = face.bbox[0] + face.bbox[2] / 2;
    const cy = face.bbox[1] + face.bbox[3] / 2;
    const raw = await predictor.predict(cropFace(face), cx, cy);
    pred = smooth(raw);
  } else {
    smoothed = null;
  }

  const decision = decide(
    {
      face,
      brightness,
      motion,
      noFaceStreak,
      uncertaintyThreshold: predictor.meta.abstain_std_threshold,
    },
    pred,
  );

  plane.render(decision.abstain ? null : pred, decision);
  updateDetails(pred, decision);
  requestAnimationFrame(loop);
}

async function start() {
  $("consent").classList.add("hidden");
  $("live").classList.remove("hidden");
  $("state-badge").textContent = "loading model…";
  await predictor.load();
  landmarker = await createLandmarker();
  stream = await startCamera(video);
  running = true;
  requestAnimationFrame(loop);
}

function stop() {
  running = false;
  stopCamera(stream, video);
  $("live").classList.add("hidden");
  $("consent").classList.remove("hidden");
}

$("start-btn").addEventListener("click", () => start().catch((e) => {
  $("state-badge").textContent = `error: ${e?.message ?? e}`;
}));
$("stop-btn").addEventListener("click", stop);

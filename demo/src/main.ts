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
import { EMOTION_DISPLAY, PRIVACY, THRESHOLDS } from "./config";
import {
  createLandmarker,
  detect,
  landmarkMotion,
  type FaceInfo,
} from "./landmarker";
import { AffectPredictor, EmotionPredictor, jointRegion, type Prediction } from "./model";
import { coarseLabel } from "./readout";
import { VAPlane } from "./viz";
import { PulseEstimator, skinRGB, type PulseEstimate } from "./physio/pulse";
import { EyeTracker, yawProxy, type EyeState } from "./physio/eyes";
import { Timeline } from "./timeline";
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
const emotionPredictor = new EmotionPredictor();

let prevLandmarks: { x: number; y: number }[] | null = null;
let noFaceStreak = 0;
let smoothed: Prediction | null = null;
let running = false;

// Body signals + timeline state.
const physioCanvas = document.createElement("canvas");
const pulseEst = new PulseEstimator();
const eyeTracker = new EyeTracker();
const timeline = new Timeline($("timeline") as HTMLCanvasElement);
const BASELINE_SECONDS = 60;
let t0 = performance.now();
const clock = () => (performance.now() - t0) / 1000;
let lastPulse: PulseEstimate = { bpm: null, quality: 0, buffered: 0 };
let lastPulseAt = 0;
let lastTimelineAt = 0;
let pulseHistory: { t: number; bpm: number }[] = [];
let pulseBaseline: number | null = null;

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

const word = (x: number, neg: string, mid: string, pos: string) =>
  x > 0.25 ? pos : x < -0.25 ? neg : mid;
const signed = (x: number) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(2);

function updateExpression(pred: Prediction | null, d: Decision) {
  const show = pred && !d.abstain;
  $("m-valence").textContent = show ? signed(pred.valence) : "—";
  $("m-arousal").textContent = show ? signed(pred.arousal) : "—";
  $("m-valence-word").textContent = show
    ? `looks ${word(pred.valence, "unpleasant", "neutral", "pleasant")}`
    : "pleasant ↔ unpleasant";
  $("m-arousal-word").textContent = show
    ? `looks ${word(pred.arousal, "calm", "neither calm nor activated", "activated")}`
    : "calm ↔ activated";
  const q = predictor.meta.joint_q;
  if (show && q) {
    const r = jointRegion(pred, q);
    $("m-region").textContent =
      `90% region: valence ${signed(r.v[0])} to ${signed(r.v[1])}, ` +
      `arousal ${signed(r.a[0])} to ${signed(r.a[1])}`;
    // Area of the square is 4; the abstention threshold is validation-calibrated.
    const share = r.area / (predictor.meta.abstain_area_threshold ?? 4);
    $("m-certainty").textContent =
      share < 0.4 ? "The model is fairly sure: the box is small."
      : share < 0.75 ? "Moderately sure: the true value could be anywhere in the box."
      : "Barely sure: the box is large; one more step and it would say “not sure”.";
  } else {
    $("m-region").textContent = "90% region: —";
    $("m-certainty").textContent = d.abstain && d.reason
      ? `Not giving an estimate: ${d.reason.toLowerCase()}.` : "";
  }
  // Illustrative label, inside the collapsed panel only.
  $("d-readout").textContent = show ? coarseLabel(pred.valence, pred.arousal) : "—";
  $("d-quality").textContent = `${Math.round(d.quality * 100)}%`;
  $("d-reason").textContent = d.reason ?? "—";
  $("d-model").textContent = predictor.label;
  const badge = $("state-badge");
  if (d.abstain) {
    badge.textContent = `Not sure: ${d.reason}`;
    badge.className = "state abstain";
  } else {
    badge.textContent = "Estimating";
    badge.className = "state ok";
  }
}

function updateBody(p: PulseEstimate, e: EyeState) {
  $("m-pulse").textContent = p.bpm !== null ? `${Math.round(p.bpm)}` : "—";
  let note: string;
  if (p.bpm !== null) {
    note = "bpm";
    if (pulseBaseline !== null && clock() > BASELINE_SECONDS) {
      const dlt = Math.round(p.bpm - pulseBaseline);
      note = `bpm · ${dlt >= 0 ? "+" : "−"}${Math.abs(dlt)} vs your first minute`;
    }
  } else if (p.buffered < 6) {
    note = `hold still… ${Math.max(0, Math.ceil(6 - p.buffered))} s`;
  } else {
    note = "signal too noisy (light / movement)";
  }
  $("m-pulse-note").textContent = note;
  ($("m-pulse-q") as HTMLElement).style.width = `${Math.round(Math.min(1, p.quality / 0.6) * 100)}%`;
  $("d-pulse-q").textContent = `${Math.round(p.quality * 100)}% of band power at the peak`;
  $("m-blinks").textContent = e.blinkRate !== null ? `${Math.round(e.blinkRate)}` : "—";
  $("m-blinks-note").textContent = e.blinkRate !== null ? "per minute (typical 10–20)" : "measuring…";
  $("m-gaze").textContent = e.onScreen !== null ? `${Math.round(e.onScreen * 100)}%` : "—";
  $("m-gaze-note").textContent = e.away ? "looking away now" : "of the last minute";
}

function updateSummary() {
  const s = timeline.summary();
  if (s.n < 10) return;
  const parts: string[] = [];
  if (s.v !== null && s.a !== null) {
    parts.push(
      `your face looked mostly ${word(s.v, "unpleasant", "neutral", "pleasant")} and ` +
        `${word(s.a, "calm", "neither calm nor activated", "activated")}`,
    );
  }
  parts.push(`the system said “not sure” ${Math.round(s.abstainShare * 100)}% of the time`);
  const bpms = pulseHistory.filter((b) => b.t >= clock() - timeline.windowSeconds).map((b) => b.bpm);
  if (bpms.length >= 3) {
    const sorted = [...bpms].sort((a, b) => a - b);
    parts.push(`pulse ${Math.round(sorted[Math.floor(sorted.length / 2)])} bpm (median)`);
  }
  const win = timeline.windowSeconds >= 120 ? `${timeline.windowSeconds / 60} minutes` : "minute";
  $("tl-summary").textContent = `Over the last ${win}: ${parts.join("; ")}.`;
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
    stdV: a * smoothed.stdV + (1 - a) * pred.stdV,
    stdA: a * smoothed.stdA + (1 - a) * pred.stdA,
  };
  return smoothed;
}

// Light temporal smoothing of emotion probabilities to reduce flicker.
let emoSmooth: Record<string, number> = {};
function renderEmotion(e: import("./model").EmotionResult | null) {
  const emojiEl = $("emotion-emoji");
  const labelEl = $("emotion-label");
  const confEl = $("emotion-conf");
  const barsEl = $("emotion-bars");
  if (!emotionPredictor.available) {
    // Model not bundled yet (still training) — keep the panel quiet.
    labelEl.textContent = "emotion model not loaded";
    confEl.textContent = "train + export emotion.onnx to enable";
    return;
  }
  if (!e) {
    emojiEl.textContent = "·";
    labelEl.textContent = "—";
    confEl.innerHTML = "&nbsp;";
    barsEl.innerHTML = "";
    emoSmooth = {};
    return;
  }
  // EMA smooth per class.
  const a = 0.6;
  for (const { label, p } of e.probs) {
    emoSmooth[label] = a * (emoSmooth[label] ?? p) + (1 - a) * p;
  }
  const ranked = Object.entries(emoSmooth)
    .map(([label, p]) => ({ label, p }))
    .sort((x, y) => y.p - x.p);
  const top = ranked[0];
  const disp = EMOTION_DISPLAY[top.label] ?? { emoji: "🙂", label: top.label };
  emojiEl.textContent = disp.emoji;
  labelEl.textContent = disp.label;
  confEl.textContent = `p ≈ ${Math.round(top.p * 100)}% (illustrative)`;
  barsEl.innerHTML = ranked
    .map((r) => {
      const d = EMOTION_DISPLAY[r.label] ?? { emoji: "", label: r.label };
      return `<div class="ebar"><span>${d.emoji} ${d.label}</span>
        <span class="etrack"><span class="efill" style="width:${Math.round(r.p * 100)}%"></span></span>
        <span class="epct">${Math.round(r.p * 100)}%</span></div>`;
    })
    .join("");
}

async function loop() {
  if (!running || !landmarker) return;
  const tsMs = performance.now();
  const t = clock();
  const face = detect(landmarker, video, tsMs);
  const brightness = frameBrightness(video, scratch);
  const motion = landmarkMotion(prevLandmarks, face.landmarks);
  prevLandmarks = face.landmarks;
  noFaceStreak = face.present ? 0 : noFaceStreak + 1;

  let pred: Prediction | null = null;
  let emotion = null;
  if (face.present && face.bbox) {
    const cx = face.bbox[0] + face.bbox[2] / 2;
    const cy = face.bbox[1] + face.bbox[3] / 2;
    const crop = cropFace(face);
    const raw = await predictor.predict(crop, cx, cy);
    pred = smooth(raw);
    emotion = await emotionPredictor.predict(crop);
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
      jointQ: predictor.meta.joint_q,
      areaThreshold: predictor.meta.abstain_area_threshold,
    },
    pred,
  );

  // --- body signals ---
  if (face.present && face.landmarks && face.bbox) {
    if (motion < THRESHOLDS.maxMotion * 0.5) {
      const rgb = sampleSkin(face);
      if (rgb) pulseEst.push({ t, rgb });
    }
  } else {
    pulseEst.reset();
  }
  const eyes = eyeTracker.update(
    t,
    face.present ? face.blendshapes ?? null : null,
    face.landmarks ? yawProxy(face.landmarks) : 0,
  );
  if (t - lastPulseAt >= 1) {
    lastPulseAt = t;
    lastPulse = pulseEst.estimate();
    if (lastPulse.bpm !== null) {
      pulseHistory.push({ t, bpm: lastPulse.bpm });
      if (t <= BASELINE_SECONDS) {
        const early = pulseHistory.map((b) => b.bpm).sort((a, b) => a - b);
        pulseBaseline = early[Math.floor(early.length / 2)];
      }
    }
    pulseHistory = pulseHistory.filter((b) => b.t >= t - 300);
  }

  // --- timeline (2 points/s) ---
  if (t - lastTimelineAt >= 0.5) {
    lastTimelineAt = t;
    const q = predictor.meta.joint_q;
    const ok = pred && !decision.abstain;
    const r = ok && q ? jointRegion(pred!, q) : null;
    timeline.push({
      t,
      v: ok ? pred!.valence : null,
      a: ok ? pred!.arousal : null,
      vLo: r?.v[0], vHi: r?.v[1], aLo: r?.a[0], aHi: r?.a[1],
      bpm: lastPulse.bpm,
    });
    timeline.render();
    updateSummary();
  }

  plane.render(decision.abstain ? null : pred, decision, predictor.meta.joint_q);
  updateExpression(pred, decision);
  updateBody(lastPulse, eyes);
  renderEmotion(decision.abstain ? null : emotion);
  requestAnimationFrame(loop);
}

/** Mean skin RGB from forehead + cheeks at 320x240 (enough for rPPG). */
function sampleSkin(face: FaceInfo): [number, number, number] | null {
  const w = 320;
  const h = 240;
  physioCanvas.width = w;
  physioCanvas.height = h;
  const ctx = physioCanvas.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(video, 0, 0, w, h);
  const { data } = ctx.getImageData(0, 0, w, h);
  return skinRGB(data, w, h, face.landmarks!, face.bbox![2]);
}

async function start() {
  $("consent").classList.add("hidden");
  $("live").classList.remove("hidden");
  $("state-badge").textContent = "loading model…";
  await predictor.load();
  await emotionPredictor.load();
  landmarker = await createLandmarker();
  stream = await startCamera(video);
  t0 = performance.now();
  pulseEst.reset();
  eyeTracker.reset();
  timeline.clear();
  pulseHistory = [];
  pulseBaseline = null;
  lastPulse = { bpm: null, quality: 0, buffered: 0 };
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
document.querySelectorAll<HTMLButtonElement>(".seg button").forEach((b) =>
  b.addEventListener("click", () => {
    timeline.windowSeconds = Number(b.dataset.win);
    document.querySelectorAll(".seg button").forEach((x) => x.classList.toggle("on", x === b));
    timeline.render();
    updateSummary();
  }),
);

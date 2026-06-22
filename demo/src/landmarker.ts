/**
 * MediaPipe FaceLandmarker wrapper (on-device). Provides:
 *  - detection presence + bounding box (for no-face / small-face abstention)
 *  - landmarks (for aligned crop + motion estimation)
 *
 * The model asset and wasm are loaded locally/CDN; no frames are uploaded.
 */
import {
  FaceLandmarker,
  FilesetResolver,
  type FaceLandmarkerResult,
} from "@mediapipe/tasks-vision";

const WASM_PATH =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

export interface FaceInfo {
  present: boolean;
  /** Normalized [0,1] bbox: x, y, w, h. */
  bbox: [number, number, number, number] | null;
  /** Normalized landmark points (subset used for crop + motion). */
  landmarks: { x: number; y: number }[] | null;
}

export async function createLandmarker(): Promise<FaceLandmarker> {
  const fileset = await FilesetResolver.forVisionTasks(WASM_PATH);
  return FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: false,
  });
}

export function detect(
  lm: FaceLandmarker,
  video: HTMLVideoElement,
  tsMs: number,
): FaceInfo {
  const res: FaceLandmarkerResult = lm.detectForVideo(video, tsMs);
  const pts = res.faceLandmarks?.[0];
  if (!pts || pts.length === 0) {
    return { present: false, bbox: null, landmarks: null };
  }
  let minX = 1,
    minY = 1,
    maxX = 0,
    maxY = 0;
  for (const p of pts) {
    minX = Math.min(minX, p.x);
    minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x);
    maxY = Math.max(maxY, p.y);
  }
  return {
    present: true,
    bbox: [minX, minY, maxX - minX, maxY - minY],
    landmarks: pts.map((p) => ({ x: p.x, y: p.y })),
  };
}

/** Mean per-landmark displacement vs the previous frame (normalized units). */
export function landmarkMotion(
  prev: { x: number; y: number }[] | null,
  cur: { x: number; y: number }[] | null,
): number {
  if (!prev || !cur || prev.length !== cur.length) return 0;
  let sum = 0;
  for (let i = 0; i < cur.length; i++) {
    sum += Math.hypot(cur[i].x - prev[i].x, cur[i].y - prev[i].y);
  }
  return sum / cur.length;
}

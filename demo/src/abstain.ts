/**
 * Abstention decision — the core of the demo's honesty. Runs BEFORE trusting any
 * model output and again AFTER, so the system says "not sure" whenever the
 * signal is unreliable, independent of model quality.
 *
 * Order matters: physical signal problems (no face, lighting, framing, motion)
 * are checked first; only if the input is trustworthy do we consult the model's
 * own predictive uncertainty for the "genuinely ambiguous" case.
 */
import {
  ABSTAIN_REASONS,
  THRESHOLDS,
  type AbstainReason,
} from "./config";
import type { FaceInfo } from "./landmarker";
import type { Prediction } from "./model";

export interface SignalContext {
  face: FaceInfo;
  brightness: number;
  motion: number;
  noFaceStreak: number;
  uncertaintyThreshold: number; // from model meta (conformal-derived if available)
}

export interface Decision {
  abstain: boolean;
  reason: AbstainReason;
  /** Confidence in [0,1], derived from predictive std (display only). */
  confidence: number;
  /** Signal-quality score in [0,1] combining lighting/framing/motion. */
  quality: number;
}

export function decide(
  ctx: SignalContext,
  pred: Prediction | null,
): Decision {
  const quality = signalQuality(ctx);

  // 1. Physical signal gates (independent of the model).
  if (!ctx.face.present) {
    if (ctx.noFaceStreak >= THRESHOLDS.noFaceGrace)
      return abstain(ABSTAIN_REASONS.NO_FACE, quality);
  }
  if (ctx.brightness < THRESHOLDS.minBrightness)
    return abstain(ABSTAIN_REASONS.LOW_LIGHT, quality);
  if (ctx.brightness > THRESHOLDS.maxBrightness)
    return abstain(ABSTAIN_REASONS.BRIGHT, quality);
  if (ctx.face.present && ctx.face.bbox) {
    const area = ctx.face.bbox[2] * ctx.face.bbox[3];
    if (area < THRESHOLDS.minFaceArea)
      return abstain(ABSTAIN_REASONS.SMALL_FACE, quality);
  }
  if (ctx.motion > THRESHOLDS.maxMotion)
    return abstain(ABSTAIN_REASONS.MOTION, quality);

  // 2. Model-driven gate: the prediction itself is too uncertain to call.
  if (!pred) return abstain(ABSTAIN_REASONS.NO_FACE, quality);
  const confidence = stdToConfidence(pred.std);
  if (pred.std > ctx.uncertaintyThreshold)
    return abstain(ABSTAIN_REASONS.AMBIGUOUS, quality, confidence);

  return { abstain: false, reason: null, confidence, quality };
}

function abstain(
  reason: AbstainReason,
  quality: number,
  confidence = 0,
): Decision {
  return { abstain: true, reason, confidence, quality };
}

/** Map predictive std to a display confidence. Monotonic, bounded [0,1]. */
function stdToConfidence(std: number): number {
  return Math.max(0, Math.min(1, 1 - std / 0.7));
}

/** Combine lighting, framing, and motion into a single 0–1 quality score. */
function signalQuality(ctx: SignalContext): number {
  if (!ctx.face.present) return 0;
  const lightOk =
    ctx.brightness >= THRESHOLDS.minBrightness &&
    ctx.brightness <= THRESHOLDS.maxBrightness
      ? 1
      : 0.3;
  const area = ctx.face.bbox ? ctx.face.bbox[2] * ctx.face.bbox[3] : 0;
  const framing = Math.max(0, Math.min(1, area / 0.15));
  const motionOk = Math.max(0, 1 - ctx.motion / THRESHOLDS.maxMotion);
  return lightOk * framing * motionOk;
}

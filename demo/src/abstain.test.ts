/**
 * Unit tests for the abstention logic — the demo's honesty guarantee. These run
 * with `npm test` (vitest) and need no browser, camera, or model.
 *
 * The key invariants: physical signal problems force abstention regardless of
 * model output, and a confident prediction with good signal is NOT abstained.
 */
import { describe, expect, it } from "vitest";
import { decide, type SignalContext } from "./abstain";
import { ABSTAIN_REASONS, THRESHOLDS } from "./config";
import type { Prediction } from "./model";

const goodFace = {
  present: true,
  bbox: [0.35, 0.3, 0.3, 0.4] as [number, number, number, number],
  landmarks: [{ x: 0.5, y: 0.5 }],
};

function ctx(over: Partial<SignalContext> = {}): SignalContext {
  return {
    face: goodFace,
    brightness: 128,
    motion: 0,
    noFaceStreak: 0,
    uncertaintyThreshold: THRESHOLDS.uncertaintyStd,
    ...over,
  };
}

const confident: Prediction = { valence: 0.3, arousal: 0.2, std: 0.1 };
const unsure: Prediction = { valence: 0.3, arousal: 0.2, std: 0.9 };

describe("abstention", () => {
  it("answers when signal is good and model is confident", () => {
    const d = decide(ctx(), confident);
    expect(d.abstain).toBe(false);
    expect(d.confidence).toBeGreaterThan(0.5);
  });

  it("abstains with NO_FACE after the grace period", () => {
    const d = decide(
      ctx({ face: { present: false, bbox: null, landmarks: null },
            noFaceStreak: THRESHOLDS.noFaceGrace + 1 }),
      confident,
    );
    expect(d.abstain).toBe(true);
    expect(d.reason).toBe(ABSTAIN_REASONS.NO_FACE);
  });

  it("abstains in low light even with a confident model", () => {
    const d = decide(ctx({ brightness: THRESHOLDS.minBrightness - 1 }), confident);
    expect(d.abstain).toBe(true);
    expect(d.reason).toBe(ABSTAIN_REASONS.LOW_LIGHT);
  });

  it("abstains when overexposed", () => {
    const d = decide(ctx({ brightness: THRESHOLDS.maxBrightness + 1 }), confident);
    expect(d.reason).toBe(ABSTAIN_REASONS.BRIGHT);
  });

  it("abstains when the face is too small", () => {
    const d = decide(
      ctx({ face: { ...goodFace, bbox: [0.4, 0.4, 0.1, 0.1] } }),
      confident,
    );
    expect(d.reason).toBe(ABSTAIN_REASONS.SMALL_FACE);
  });

  it("abstains under excessive motion", () => {
    const d = decide(ctx({ motion: THRESHOLDS.maxMotion + 0.01 }), confident);
    expect(d.reason).toBe(ABSTAIN_REASONS.MOTION);
  });

  it("abstains as AMBIGUOUS when good signal but uncertain model", () => {
    const d = decide(ctx(), unsure);
    expect(d.abstain).toBe(true);
    expect(d.reason).toBe(ABSTAIN_REASONS.AMBIGUOUS);
  });

  it("ranks confidence monotonically with predictive std", () => {
    const a = decide(ctx(), { ...confident, std: 0.1 }).confidence;
    const b = decide(ctx(), { ...confident, std: 0.3 }).confidence;
    expect(a).toBeGreaterThan(b);
  });

  it("reports lower quality when framing/lighting degrade", () => {
    const good = decide(ctx(), confident).quality;
    const bad = decide(ctx({ motion: THRESHOLDS.maxMotion * 0.9 }), confident).quality;
    expect(good).toBeGreaterThan(bad);
  });
});

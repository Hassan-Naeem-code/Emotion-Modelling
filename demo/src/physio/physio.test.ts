import { describe, expect, it } from "vitest";
import { PulseEstimator } from "./pulse";
import { EyeTracker, yawProxy } from "./eyes";

/** Seeded RNG so the tests are deterministic. */
function rng(seed: number) {
  return () => {
    seed = (seed * 1664525 + 1013904223) % 4294967296;
    return seed / 4294967296;
  };
}

/** Skin-coloured trace with a pulse at `bpm`, camera noise and jittered frame times. */
function skinTrace(bpm: number, seconds: number, noise: number, seed = 1) {
  const r = rng(seed);
  const est = new PulseEstimator();
  let t = 0;
  while (t < seconds) {
    const p = Math.sin(2 * Math.PI * (bpm / 60) * t);
    // Pulsatile change is strongest in green, as in real skin.
    const rgb: [number, number, number] = [
      180 + 0.3 * p + noise * (r() - 0.5),
      120 + 0.6 * p + noise * (r() - 0.5),
      100 + 0.2 * p + noise * (r() - 0.5),
    ];
    est.push({ t, rgb });
    t += 1 / 30 + (r() - 0.5) * 0.01; // ~30 fps with jitter
  }
  return est;
}

describe("pulse (rPPG / POS)", () => {
  it.each([60, 72, 95, 130])("recovers %i bpm from a clean skin trace", (bpm) => {
    const e = skinTrace(bpm, 10, 0.4).estimate();
    expect(e.bpm).not.toBeNull();
    expect(Math.abs(e.bpm! - bpm)).toBeLessThan(3);
    expect(e.quality).toBeGreaterThan(0.4);
  });

  it("stays blank on pure noise instead of guessing", () => {
    const r = rng(7);
    const est = new PulseEstimator();
    for (let t = 0; t < 10; t += 1 / 30) {
      est.push({ t, rgb: [180 + 4 * (r() - 0.5), 120 + 4 * (r() - 0.5), 100 + 4 * (r() - 0.5)] });
    }
    expect(est.estimate().bpm).toBeNull();
  });

  it("waits for enough signal before estimating", () => {
    const e = skinTrace(72, 3, 0.4).estimate();
    expect(e.bpm).toBeNull();
    expect(e.buffered).toBeLessThan(6);
  });
});

describe("eyes", () => {
  const open = { eyeBlinkLeft: 0.05, eyeBlinkRight: 0.05 };
  const shut = { eyeBlinkLeft: 0.9, eyeBlinkRight: 0.9 };

  it("counts blinks per minute", () => {
    const tr = new EyeTracker();
    let st;
    // 30 s at 30 fps, one 150 ms blink every 4 s -> 15 blinks/min.
    for (let i = 0; i < 900; i++) {
      const t = i / 30;
      const inBlink = t % 4 > 2 && t % 4 < 2.15;
      st = tr.update(t, inBlink ? shut : open, 0);
    }
    expect(st!.blinkRate).not.toBeNull();
    expect(Math.abs(st!.blinkRate! - 15)).toBeLessThan(2.5);
  });

  it("does not count eyes held shut as blinks", () => {
    const tr = new EyeTracker();
    let st;
    for (let i = 0; i < 600; i++) {
      const t = i / 30;
      st = tr.update(t, t > 5 && t < 8 ? shut : open, 0);
    }
    expect(st!.blinkRate).toBe(0);
  });

  it("measures time looking at the screen", () => {
    const tr = new EyeTracker();
    let st;
    for (let i = 0; i < 600; i++) {
      const t = i / 30;
      const away = i % 4 === 0; // 25% of frames looking far right
      st = tr.update(t, away ? { ...open, eyeLookOutRight: 0.8 } : open, 0);
    }
    expect(Math.abs(st!.onScreen! - 0.75)).toBeLessThan(0.02);
  });

  it("treats a turned head as looking away", () => {
    const lm = Array.from({ length: 300 }, () => ({ x: 0.5, y: 0.5 }));
    lm[33] = { x: 0.4, y: 0.5 };
    lm[263] = { x: 0.6, y: 0.5 };
    lm[1] = { x: 0.5, y: 0.6 };
    expect(Math.abs(yawProxy(lm))).toBeLessThan(0.05);
    lm[1] = { x: 0.58, y: 0.6 };
    expect(yawProxy(lm)).toBeGreaterThan(0.35);
    const tr = new EyeTracker();
    expect(tr.update(0, open, yawProxy(lm)).away).toBe(true);
  });
});

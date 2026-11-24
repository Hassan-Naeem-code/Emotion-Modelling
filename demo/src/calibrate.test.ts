import { describe, expect, it } from "vitest";
import {
  K, conformalQuantile, contains, fitPersonal, personalRegion, pxToVA, type Sample,
} from "./calibrate";
import { jointRegion, type Prediction } from "./model";

// Small seeded RNG so the tests are deterministic.
function rng(seed: number) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32);
}
function normal(r: () => number) {
  const u = Math.max(r(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
}

/** A user whose labels sit at a fixed offset from the model, plus noise. */
function tap(r: () => number, off: [number, number], sd = 0.1): Sample {
  const v = r() * 1.2 - 0.6, a = r() * 1.2 - 0.6;
  // Real taps land on the chart, so labels live in [-1, 1].
  const c = (x: number) => Math.max(-1, Math.min(1, x));
  return { v, a, sv: sd, sa: sd, tv: c(v + off[0] + sd * normal(r)), ta: c(a + off[1] + sd * normal(r)) };
}
const pred = (s: Sample): Prediction =>
  ({ valence: s.v, arousal: s.a, std: s.sv, stdV: s.sv, stdA: s.sa });

describe("conformalQuantile", () => {
  it("takes the ceil((n+1)(1-alpha))-th smallest score", () => {
    const xs = [5, 1, 4, 2, 3, 10, 9, 8, 7, 6];        // n = 10 -> rank 10
    expect(conformalQuantile(xs)).toBe(10);
    expect(conformalQuantile(xs.slice(0, 8))).toBe(Infinity);  // n = 8 is too few
  });
});

describe("fitPersonal", () => {
  it("needs K taps", () => {
    const r = rng(1);
    expect(fitPersonal(Array.from({ length: K - 1 }, () => tap(r, [0, 0])))).toBeNull();
  });

  it("recovers the user's offset", () => {
    const r = rng(2);
    const c = fitPersonal(Array.from({ length: K }, () => tap(r, [0.4, -0.3])))!;
    expect(c.offV).toBeCloseTo(0.4, 1);
    expect(c.offA).toBeCloseTo(-0.3, 1);
    expect(Number.isFinite(c.q)).toBe(true);
  });

  it("covers new taps at least 90% of the time on average, where the original box does not", () => {
    const r = rng(3);
    let hit = 0, hitOrig = 0, total = 0;
    for (let trial = 0; trial < 300; trial++) {
      const c = fitPersonal(Array.from({ length: K }, () => tap(r, [0.4, -0.3])))!;
      for (let j = 0; j < 20; j++) {
        const s = tap(r, [0.4, -0.3]);
        hit += +contains(personalRegion(pred(s), c), s.tv, s.ta);
        hitOrig += +contains(jointRegion(pred(s), 2.46), s.tv, s.ta);
        total++;
      }
    }
    expect(hit / total).toBeGreaterThan(0.9);
    expect(hitOrig / total).toBeLessThan(0.3);
  });
});

describe("pxToVA", () => {
  it("maps the canvas corners and centre", () => {
    expect(pxToVA(0, 0, 360, 360)).toEqual([-1, 1]);
    expect(pxToVA(360, 360, 360, 360)).toEqual([1, -1]);
    expect(pxToVA(180, 180, 360, 360)).toEqual([0, 0]);
  });
});

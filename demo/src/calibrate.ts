/**
 * Personal calibration: the paper's remedy for a face the calibration data
 * did not cover (Section "Twenty labelled examples restore the guarantee").
 *
 * The user taps, K times, the point on the VA plane that matches the
 * expression they are making. Each tap pairs the model's prediction (mean and
 * per-axis std, as shown) with the user's own label. Then, as in the paper's
 * "+shift" recalibration:
 *   - half the taps estimate the user's mean offset (label minus prediction);
 *   - the other half set the max-score threshold on the offset-corrected,
 *     std-normalised residuals.
 * The halves are disjoint, so on future taps that are exchangeable with these
 * (same person, same kind of expressions, labelled the same way) the personal
 * box contains the user's label with probability at least 1 - alpha.
 *
 * Pure functions only: no DOM, no storage. Tested in calibrate.test.ts.
 */
import type { Prediction, Region } from "./model";

export const K = 20;          // taps needed; 10 per half (alpha = 0.1 needs >= 9)
export const ALPHA = 0.1;

/** One tap: what the model showed, and where the user said the truth is. */
export interface Sample {
  v: number; a: number;       // predicted valence / arousal
  sv: number; sa: number;     // predicted per-axis std
  tv: number; ta: number;     // the user's label
}

export interface Personal {
  offV: number; offA: number; // mean label - prediction, from the first half
  q: number;                  // max-score threshold, from the second half
  n: number;                  // taps used
}

/** ceil((n+1)(1-alpha))-th smallest score; Infinity if there are too few. */
export function conformalQuantile(scores: number[], alpha = ALPHA): number {
  const n = scores.length;
  const r = Math.ceil((n + 1) * (1 - alpha));
  if (r > n) return Infinity;
  return [...scores].sort((x, y) => x - y)[r - 1];
}

const EPS = 1e-6;
const score = (s: Sample, offV: number, offA: number) =>
  Math.max(Math.abs(s.tv - s.v - offV) / (s.sv + EPS), Math.abs(s.ta - s.a - offA) / (s.sa + EPS));

/** Fit from the first K taps; null until there are K. Halves alternate
 *  (even taps -> offset, odd taps -> threshold) so both span the session. */
export function fitPersonal(samples: Sample[], alpha = ALPHA): Personal | null {
  if (samples.length < K) return null;
  const use = samples.slice(0, K);
  const first = use.filter((_, i) => i % 2 === 0);
  const second = use.filter((_, i) => i % 2 === 1);
  const offV = mean(first.map((s) => s.tv - s.v));
  const offA = mean(first.map((s) => s.ta - s.a));
  const q = conformalQuantile(second.map((s) => score(s, offV, offA)), alpha);
  return { offV, offA, q, n: use.length };
}

/** The personal box for a new prediction, clipped to the VA square. */
export function personalRegion(p: Prediction, c: Personal): Region {
  const cv = p.valence + c.offV;
  const ca = p.arousal + c.offA;
  const v: [number, number] = [clamp(cv - c.q * p.stdV), clamp(cv + c.q * p.stdV)];
  const a: [number, number] = [clamp(ca - c.q * p.stdA), clamp(ca + c.q * p.stdA)];
  return { v, a, area: (v[1] - v[0]) * (a[1] - a[0]) };
}

export function contains(r: Region, tv: number, ta: number): boolean {
  return tv >= r.v[0] && tv <= r.v[1] && ta >= r.a[0] && ta <= r.a[1];
}

/** Canvas pixel -> VA (valence right, arousal up), clipped to [-1, 1]. */
export function pxToVA(x: number, y: number, width: number, height: number): [number, number] {
  return [clamp((x / width) * 2 - 1), clamp(1 - (y / height) * 2)];
}

const mean = (xs: number[]) => xs.reduce((s, x) => s + x, 0) / xs.length;
const clamp = (x: number) => Math.max(-1, Math.min(1, x));

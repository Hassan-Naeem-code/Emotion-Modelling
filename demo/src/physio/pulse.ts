/**
 * Heart rate from the face (remote photoplethysmography, rPPG).
 *
 * Each heartbeat pushes blood into the skin and changes its colour very slightly.
 * We average the RGB of three skin patches (forehead, both cheeks) every frame,
 * and recover the pulse with the POS algorithm (Wang et al., 2017), the same
 * method as models/rppg.py: per 1.6 s window, normalise each channel by its mean,
 * project onto the plane orthogonal to skin tone, combine the two projections
 * with an adaptive weight, and overlap-add. The heart rate is the strongest
 * frequency between 42 and 210 bpm over the last ~10 s.
 *
 * HONESTY: rPPG breaks under motion, changing light and low resolution. Every
 * estimate carries a quality score (share of in-band power at the peak), and the
 * UI shows a number only when quality clears a threshold. It is a physiological
 * arousal cue, never a reading of how someone feels.
 */

export interface PulseSample {
  t: number; // seconds
  rgb: [number, number, number];
}

export interface PulseEstimate {
  /** Beats per minute, or null while the signal is not usable. */
  bpm: number | null;
  /** 0..1: fraction of 0.7-3.5 Hz power concentrated at the peak. */
  quality: number;
  /** Seconds of clean signal buffered (needs PULSE.minSeconds). */
  buffered: number;
}

export const PULSE = {
  windowSeconds: 10,
  minSeconds: 6,
  fs: 30, // resampling rate (Hz)
  loHz: 0.7, // 42 bpm
  hiHz: 3.5, // 210 bpm
  minQuality: 0.4, // pure camera noise reaches ~0.3; a real pulse 0.7+
  posWindow: 1.6, // seconds
};

export class PulseEstimator {
  private buf: PulseSample[] = [];

  push(s: PulseSample): void {
    this.buf.push(s);
    const cutoff = s.t - PULSE.windowSeconds;
    while (this.buf.length && this.buf[0].t < cutoff) this.buf.shift();
  }

  /** Motion or a lost face breaks the colour trace: start over. */
  reset(): void {
    this.buf = [];
  }

  estimate(): PulseEstimate {
    const n = this.buf.length;
    const buffered = n > 1 ? this.buf[n - 1].t - this.buf[0].t : 0;
    if (buffered < PULSE.minSeconds || n < 30) return { bpm: null, quality: 0, buffered };
    const rgb = resample(this.buf, PULSE.fs);
    const h = pos(rgb, Math.round(PULSE.posWindow * PULSE.fs));
    const { freq, quality } = dominantFrequency(detrend(h, PULSE.fs), PULSE.fs);
    const ok = quality >= PULSE.minQuality;
    return { bpm: ok ? freq * 60 : null, quality, buffered };
  }
}

/** Linear interpolation of an irregularly sampled RGB trace onto a uniform grid. */
export function resample(buf: PulseSample[], fs: number): number[][] {
  const t0 = buf[0].t;
  const t1 = buf[buf.length - 1].t;
  const out: number[][] = [];
  let j = 0;
  for (let t = t0; t <= t1; t += 1 / fs) {
    while (j < buf.length - 2 && buf[j + 1].t < t) j++;
    const a = buf[j];
    const b = buf[j + 1];
    const w = b.t > a.t ? (t - a.t) / (b.t - a.t) : 0;
    out.push([0, 1, 2].map((c) => a.rgb[c] + w * (b.rgb[c] - a.rgb[c])));
  }
  return out;
}

/** POS projection with overlap-add (Wang et al., 2017, Algorithm 1). */
export function pos(rgb: number[][], l: number): number[] {
  const n = rgb.length;
  const h = new Array(n).fill(0);
  for (let m = 0; m + l <= n; m++) {
    const mean = [0, 1, 2].map((c) => {
      let s = 0;
      for (let k = m; k < m + l; k++) s += rgb[k][c];
      return s / l || 1;
    });
    const s1: number[] = [];
    const s2: number[] = [];
    for (let k = m; k < m + l; k++) {
      const r = rgb[k][0] / mean[0];
      const g = rgb[k][1] / mean[1];
      const b = rgb[k][2] / mean[2];
      s1.push(g - b);
      s2.push(-2 * r + g + b);
    }
    const alpha = std(s1) / (std(s2) || 1);
    const seg = s1.map((v, i) => v + alpha * s2[i]);
    const mu = seg.reduce((a, b) => a + b, 0) / l;
    for (let i = 0; i < l; i++) h[m + i] += seg[i] - mu;
  }
  return h;
}

/** Remove slow drift (lighting, auto-exposure) with a 1 s moving average. */
export function detrend(x: number[], fs: number): number[] {
  const w = Math.max(1, Math.round(fs));
  return x.map((_, i) => {
    let s = 0;
    let c = 0;
    for (let k = Math.max(0, i - w); k <= Math.min(x.length - 1, i + w); k++) {
      s += x[k];
      c++;
    }
    return x[i] - s / c;
  });
}

/** Peak of the power spectrum in the heart-rate band, by direct DFT. */
export function dominantFrequency(x: number[], fs: number): { freq: number; quality: number } {
  const n = x.length;
  // Hann window reduces leakage from the window edges.
  const xw = x.map((v, i) => v * (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1))));
  let best = 0;
  let bestF = 0;
  let total = 0;
  const powers: [number, number][] = [];
  for (let f = PULSE.loHz; f <= PULSE.hiHz; f += 0.01) {
    let re = 0;
    let im = 0;
    const w = (2 * Math.PI * f) / fs;
    for (let i = 0; i < n; i++) {
      re += xw[i] * Math.cos(w * i);
      im -= xw[i] * Math.sin(w * i);
    }
    const p = re * re + im * im;
    powers.push([f, p]);
    total += p;
    if (p > best) {
      best = p;
      bestF = f;
    }
  }
  // Quality: power within ±0.1 Hz of the peak (and its first harmonic) over all
  // in-band power. A clean pulse concentrates there; noise spreads out.
  let peak = 0;
  for (const [f, p] of powers) {
    if (Math.abs(f - bestF) <= 0.1 || Math.abs(f - 2 * bestF) <= 0.1) peak += p;
  }
  return { freq: bestF, quality: total > 0 ? peak / total : 0 };
}

function std(a: number[]): number {
  const m = a.reduce((x, y) => x + y, 0) / a.length;
  return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / a.length);
}

/**
 * Mean RGB of the forehead and both cheeks, from MediaPipe face-mesh landmarks.
 * `px` is the RGBA buffer of the frame at width w and height h.
 */
const FOREHEAD = [10, 151]; // top-of-forehead and mid-forehead
const CHEEKS = [50, 280]; // left and right cheek centres

export function skinRGB(
  px: Uint8ClampedArray,
  w: number,
  h: number,
  lm: { x: number; y: number }[],
  faceW: number,
): [number, number, number] | null {
  const patches: [number, number, number][] = [];
  const fx = (lm[FOREHEAD[0]].x + lm[FOREHEAD[1]].x) / 2;
  const fy = (lm[FOREHEAD[0]].y + lm[FOREHEAD[1]].y) / 2;
  patches.push([fx, fy, faceW * 0.12]);
  for (const i of CHEEKS) patches.push([lm[i].x, lm[i].y, faceW * 0.08]);
  let r = 0;
  let g = 0;
  let b = 0;
  let c = 0;
  for (const [cx, cy, half] of patches) {
    const x0 = Math.max(0, Math.round((cx - half) * w));
    const x1 = Math.min(w - 1, Math.round((cx + half) * w));
    const y0 = Math.max(0, Math.round((cy - half * 0.6) * h));
    const y1 = Math.min(h - 1, Math.round((cy + half * 0.6) * h));
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const k = (y * w + x) * 4;
        r += px[k];
        g += px[k + 1];
        b += px[k + 2];
        c++;
      }
    }
  }
  return c > 20 ? [r / c, g / c, b / c] : null;
}

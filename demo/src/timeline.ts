/**
 * Session timeline: valence and arousal with their calibrated 90% band, and the
 * pulse, over the last 1/2/5 minutes. Gaps mark moments the system abstained, so
 * "not sure" stays visible in the history instead of being smoothed over.
 */

export interface Point {
  t: number; // seconds since start
  v: number | null; // null = abstained / no estimate
  a: number | null;
  vLo?: number;
  vHi?: number;
  aLo?: number;
  aHi?: number;
  bpm: number | null;
}

const MAX_SECONDS = 300;

export class Timeline {
  private pts: Point[] = [];
  windowSeconds = 120;
  private ctx: CanvasRenderingContext2D;

  constructor(private canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext("2d")!;
  }

  push(p: Point): void {
    this.pts.push(p);
    const cut = p.t - MAX_SECONDS;
    while (this.pts.length && this.pts[0].t < cut) this.pts.shift();
  }

  clear(): void {
    this.pts = [];
  }

  /** Summary over the visible window, for the text readout. */
  summary(): { v: number | null; a: number | null; abstainShare: number; n: number } {
    const now = this.pts[this.pts.length - 1]?.t ?? 0;
    const w = this.pts.filter((p) => p.t >= now - this.windowSeconds);
    const ok = w.filter((p) => p.v !== null);
    const mean = (k: "v" | "a") =>
      ok.length ? ok.reduce((s, p) => s + (p[k] as number), 0) / ok.length : null;
    return { v: mean("v"), a: mean("a"), abstainShare: w.length ? 1 - ok.length / w.length : 0, n: w.length };
  }

  render(): void {
    const c = this.ctx;
    const dpr = window.devicePixelRatio || 1;
    const W = this.canvas.clientWidth;
    const H = this.canvas.clientHeight;
    if (this.canvas.width !== W * dpr) {
      this.canvas.width = W * dpr;
      this.canvas.height = H * dpr;
    }
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, W, H);
    const css = getComputedStyle(document.documentElement);
    const col = (n: string) => css.getPropertyValue(n).trim();
    const now = this.pts[this.pts.length - 1]?.t ?? 0;
    const t0 = now - this.windowSeconds;
    const L = 64;
    const R = 8;
    const x = (t: number) => L + ((t - t0) / this.windowSeconds) * (W - L - R);
    const rows: { label: string; key: "v" | "a" | "bpm"; lo: number; hi: number; color: string }[] = [
      { label: "Valence", key: "v", lo: -1, hi: 1, color: col("--accent") },
      { label: "Arousal", key: "a", lo: -1, hi: 1, color: col("--accent2") },
      { label: "Pulse", key: "bpm", lo: 50, hi: 120, color: col("--pulse") },
    ];
    const gap = 10;
    const rh = (H - gap * (rows.length - 1) - 16) / rows.length;
    const vis = this.pts.filter((p) => p.t >= t0);

    rows.forEach((row, i) => {
      const top = i * (rh + gap);
      let lo = row.lo;
      let hi = row.hi;
      if (row.key === "bpm") {
        const b = vis.map((p) => p.bpm).filter((v): v is number => v !== null);
        if (b.length) {
          lo = Math.min(lo, Math.floor(Math.min(...b) / 10) * 10);
          hi = Math.max(hi, Math.ceil(Math.max(...b) / 10) * 10);
        }
      }
      const y = (v: number) => top + (1 - (v - lo) / (hi - lo)) * rh;
      c.strokeStyle = col("--border");
      c.lineWidth = 1;
      c.strokeRect(L, top, W - L - R, rh);
      if (row.key !== "bpm") {
        c.beginPath();
        c.moveTo(L, y(0));
        c.lineTo(W - R, y(0));
        c.setLineDash([3, 4]);
        c.stroke();
        c.setLineDash([]);
      }
      c.fillStyle = col("--muted");
      c.font = "12px system-ui, sans-serif";
      c.fillText(row.label, 6, top + 14);
      c.font = "10px system-ui, sans-serif";
      c.fillText(row.key === "bpm" ? `${hi}` : "+1", 6, top + 28);
      c.fillText(row.key === "bpm" ? `${lo}` : "−1", 6, top + rh - 2);

      // 90% band (valence/arousal only), drawn as contiguous segments.
      if (row.key !== "bpm") {
        const loK = row.key === "v" ? "vLo" : "aLo";
        const hiK = row.key === "v" ? "vHi" : "aHi";
        c.fillStyle = row.color + "33";
        for (const seg of segments(vis, (p) => p[row.key] !== null && p[loK] !== undefined)) {
          c.beginPath();
          seg.forEach((p, k) => (k ? c.lineTo(x(p.t), y(p[hiK]!)) : c.moveTo(x(p.t), y(p[hiK]!))));
          for (let k = seg.length - 1; k >= 0; k--) c.lineTo(x(seg[k].t), y(seg[k][loK]!));
          c.closePath();
          c.fill();
        }
      }
      c.strokeStyle = row.color;
      c.lineWidth = 2;
      for (const seg of segments(vis, (p) => p[row.key] !== null)) {
        c.beginPath();
        seg.forEach((p, k) => {
          const yy = y(Math.max(lo, Math.min(hi, p[row.key] as number)));
          if (k) c.lineTo(x(p.t), yy);
          else c.moveTo(x(p.t), yy);
        });
        c.stroke();
      }
    });
    // Time axis.
    c.fillStyle = col("--muted");
    c.font = "10px system-ui, sans-serif";
    const step = this.windowSeconds <= 60 ? 15 : this.windowSeconds <= 120 ? 30 : 60;
    for (let s = 0; s <= this.windowSeconds; s += step) {
      const xx = x(now - s);
      c.fillText(s === 0 ? "now" : `−${s}s`, xx - (s === 0 ? 18 : 10), H - 2);
    }
  }
}

/** Split points into runs where `ok` holds (gaps break the line). */
function segments(pts: Point[], ok: (p: Point) => boolean): Point[][] {
  const out: Point[][] = [];
  let cur: Point[] = [];
  let last = -Infinity;
  for (const p of pts) {
    if (ok(p) && p.t - last < 1.5) cur.push(p);
    else {
      if (cur.length > 1) out.push(cur);
      cur = ok(p) ? [p] : [];
    }
    last = p.t;
  }
  if (cur.length > 1) out.push(cur);
  return out;
}

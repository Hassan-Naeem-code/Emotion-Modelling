/**
 * Valence–Arousal plane renderer. Plots the current estimate as a dot with a
 * translucent halo whose radius encodes predictive uncertainty (small = sure,
 * big = unsure). When abstaining, the dot is replaced by an explicit
 * "Not sure" state showing the reason — the visual contract of the demo.
 */
import type { Decision } from "./abstain";
import { jointRegion, type Prediction, type Region } from "./model";

export class VAPlane {
  private ctx: CanvasRenderingContext2D;
  private size: number;

  constructor(canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext("2d")!;
    this.size = canvas.width;
  }

  private toPx(v: number, axis: "x" | "y"): number {
    // valence -> x (right = positive), arousal -> y (up = positive).
    const t = (v + 1) / 2;
    return axis === "x" ? t * this.size : (1 - t) * this.size;
  }

  private drawAxes() {
    const c = this.ctx;
    const s = this.size;
    c.clearRect(0, 0, s, s);
    c.fillStyle = "#0d1117";
    c.fillRect(0, 0, s, s);
    c.strokeStyle = "#30363d";
    c.lineWidth = 1;
    c.beginPath();
    c.moveTo(s / 2, 0);
    c.lineTo(s / 2, s);
    c.moveTo(0, s / 2);
    c.lineTo(s, s / 2);
    c.stroke();
    c.fillStyle = "#8b949e";
    c.font = "11px system-ui, sans-serif";
    c.fillText("valence +", s - 70, s / 2 - 6);
    c.fillText("valence −", 6, s / 2 - 6);
    c.fillText("arousal +", s / 2 + 6, 14);
    c.fillText("arousal −", s / 2 + 6, s - 8);
  }

  /** `personal`: the user's own calibrated box (drawn solid green), with the
   *  general box then drawn dashed for comparison. `hint`: a line of text for
   *  calibration mode. */
  render(pred: Prediction | null, decision: Decision, jointQ?: number,
         personal?: Region | null, hint?: string) {
    this.drawAxes();
    const c = this.ctx;
    const s = this.size;

    if (decision.abstain || !pred) {
      // Explicit "not sure" state — no point plotted.
      c.fillStyle = "rgba(210, 153, 34, 0.15)";
      c.fillRect(0, 0, s, s);
      c.fillStyle = "#d29922";
      c.font = "bold 18px system-ui, sans-serif";
      c.textAlign = "center";
      c.fillText("Not sure", s / 2, s / 2 - 8);
      c.font = "13px system-ui, sans-serif";
      c.fillStyle = "#e6a23c";
      c.fillText(
        "signal too weak to call",
        s / 2,
        s / 2 + 12,
      );
      if (decision.reason) {
        c.fillStyle = "#8b949e";
        c.fillText(`(${decision.reason})`, s / 2, s / 2 + 32);
      }
      c.textAlign = "left";
      this.drawHint(hint);
      return;
    }

    const x = this.toPx(pred.valence, "x");
    const y = this.toPx(pred.arousal, "y");
    if (jointQ) {
      // The calibrated 90% joint region (paper, Prop. 1): a rectangle.
      const r = jointRegion(pred, jointQ);
      const x0 = this.toPx(r.v[0], "x");
      const x1 = this.toPx(r.v[1], "x");
      const y0 = this.toPx(r.a[1], "y");
      const y1 = this.toPx(r.a[0], "y");
      if (personal) {
        // General box dashed, personal box solid: the comparison is the point.
        c.setLineDash([5, 4]);
        c.strokeStyle = "rgba(88, 166, 255, 0.55)";
        c.lineWidth = 1.2;
        c.strokeRect(x0, y0, x1 - x0, y1 - y0);
        c.setLineDash([]);
        const px0 = this.toPx(personal.v[0], "x");
        const px1 = this.toPx(personal.v[1], "x");
        const py0 = this.toPx(personal.a[1], "y");
        const py1 = this.toPx(personal.a[0], "y");
        c.fillStyle = "rgba(63, 185, 80, 0.16)";
        c.fillRect(px0, py0, px1 - px0, py1 - py0);
        c.strokeStyle = "rgba(63, 185, 80, 0.85)";
        c.lineWidth = 1.8;
        c.strokeRect(px0, py0, px1 - px0, py1 - py0);
        c.fillStyle = "#3fb950";
        c.font = "11px system-ui, sans-serif";
        c.fillText("your 90% box", Math.min(px0 + 4, s - 80), Math.max(py0 - 4, 12));
        // Where the personal correction moves the estimate.
        const cx = (px0 + px1) / 2, cy = (py0 + py1) / 2;
        c.beginPath();
        c.arc(cx, cy, 4, 0, Math.PI * 2);
        c.fill();
      } else {
        c.fillStyle = "rgba(88, 166, 255, 0.16)";
        c.fillRect(x0, y0, x1 - x0, y1 - y0);
        c.strokeStyle = "rgba(88, 166, 255, 0.6)";
        c.lineWidth = 1.5;
        c.strokeRect(x0, y0, x1 - x0, y1 - y0);
        c.fillStyle = "#8b949e";
        c.font = "11px system-ui, sans-serif";
        c.fillText("90% region", Math.min(x0 + 4, s - 70), Math.max(y0 - 4, 12));
      }
    } else {
      // Halo radius scales with predictive std (clamped for legibility).
      const halo = Math.max(6, Math.min(s / 2, pred.std * s * 0.9));
      c.beginPath();
      c.arc(x, y, halo, 0, Math.PI * 2);
      c.fillStyle = "rgba(88, 166, 255, 0.18)";
      c.fill();
      c.strokeStyle = "rgba(88, 166, 255, 0.5)";
      c.stroke();
    }

    c.beginPath();
    c.arc(x, y, 6, 0, Math.PI * 2);
    c.fillStyle = "#58a6ff";
    c.fill();
    this.drawHint(hint);
  }

  /** Mark a point the user tapped (calibration feedback). */
  mark(v: number, a: number, hit: boolean | null) {
    const c = this.ctx;
    const x = this.toPx(v, "x"), y = this.toPx(a, "y");
    c.strokeStyle = hit === null ? "#f0f6fc" : hit ? "#3fb950" : "#f85149";
    c.lineWidth = 2;
    c.beginPath();
    c.moveTo(x - 6, y - 6); c.lineTo(x + 6, y + 6);
    c.moveTo(x + 6, y - 6); c.lineTo(x - 6, y + 6);
    c.stroke();
  }

  private drawHint(hint?: string) {
    if (!hint) return;
    const c = this.ctx;
    c.fillStyle = "rgba(13, 17, 23, 0.8)";
    c.fillRect(0, this.size - 24, this.size, 24);
    c.fillStyle = "#f0f6fc";
    c.font = "12px system-ui, sans-serif";
    c.textAlign = "center";
    c.fillText(hint, this.size / 2, this.size - 8);
    c.textAlign = "left";
  }
}

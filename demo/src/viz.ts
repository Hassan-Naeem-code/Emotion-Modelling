/**
 * Valence–Arousal plane renderer. Plots the current estimate as a dot with a
 * translucent halo whose radius encodes predictive uncertainty (small = sure,
 * big = unsure). When abstaining, the dot is replaced by an explicit
 * "Not sure" state showing the reason — the visual contract of the demo.
 */
import type { Decision } from "./abstain";
import type { Prediction } from "./model";

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

  render(pred: Prediction | null, decision: Decision) {
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
      return;
    }

    const x = this.toPx(pred.valence, "x");
    const y = this.toPx(pred.arousal, "y");
    // Halo radius scales with predictive std (clamped for legibility).
    const halo = Math.max(6, Math.min(s / 2, pred.std * s * 0.9));

    c.beginPath();
    c.arc(x, y, halo, 0, Math.PI * 2);
    c.fillStyle = "rgba(88, 166, 255, 0.18)";
    c.fill();
    c.strokeStyle = "rgba(88, 166, 255, 0.5)";
    c.stroke();

    c.beginPath();
    c.arc(x, y, 6, 0, Math.PI * 2);
    c.fillStyle = "#58a6ff";
    c.fill();
  }
}

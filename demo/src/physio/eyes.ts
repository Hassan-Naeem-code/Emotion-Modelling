/**
 * Blinks and gaze from MediaPipe's face blendshapes (scores in 0..1 for
 * eyeBlinkLeft/Right and eyeLook{In,Out,Up,Down}{Left,Right}).
 *
 * Blink rate rises with fatigue and falls with concentrated visual attention; the
 * share of time looking at the screen is a simple attention cue. Both are
 * behavioural measurements, not readings of a feeling.
 */

export type Blendshapes = Record<string, number>;

export const EYES = {
  closed: 0.5, // eyeBlink score above which an eye counts as closed
  open: 0.35, // ... and below which it counts as open again (hysteresis)
  minBlinkMs: 50,
  maxBlinkMs: 500, // longer closures are eyes shut, not blinks
  lookAway: 0.55, // any eyeLook* score above this = gaze off the screen
  maxYaw: 0.35, // |head yaw| proxy above this = head turned away
  windowSeconds: 60,
};

export interface EyeState {
  /** Blinks per minute over the window, or null until 15 s are buffered. */
  blinkRate: number | null;
  /** Fraction of tracked time spent looking at the screen, or null. */
  onScreen: number | null;
  /** Current frame: eyes closed / looking away. */
  closed: boolean;
  away: boolean;
}

export class EyeTracker {
  private closedSince: number | null = null;
  private blinks: number[] = [];
  private gaze: { t: number; on: boolean }[] = [];
  private start: number | null = null;

  /**
   * @param t      timestamp in seconds
   * @param bs     blendshape scores for this frame, or null when no face
   * @param yaw    head-yaw proxy in [-1, 1] (0 = facing the camera)
   */
  update(t: number, bs: Blendshapes | null, yaw: number): EyeState {
    if (this.start === null) this.start = t;
    const cut = t - EYES.windowSeconds;
    this.blinks = this.blinks.filter((b) => b >= cut);
    this.gaze = this.gaze.filter((g) => g.t >= cut);
    if (!bs) {
      this.closedSince = null;
      this.gaze.push({ t, on: false });
      return this.state(t, false, true);
    }
    const blink = ((bs.eyeBlinkLeft ?? 0) + (bs.eyeBlinkRight ?? 0)) / 2;
    if (this.closedSince === null && blink > EYES.closed) {
      this.closedSince = t;
    } else if (this.closedSince !== null && blink < EYES.open) {
      const ms = (t - this.closedSince) * 1000;
      if (ms >= EYES.minBlinkMs && ms <= EYES.maxBlinkMs) this.blinks.push(t);
      this.closedSince = null;
    }
    const look = Math.max(
      ...["In", "Out", "Up", "Down"].flatMap((d) => [
        bs[`eyeLook${d}Left`] ?? 0,
        bs[`eyeLook${d}Right`] ?? 0,
      ]),
    );
    const away = look > EYES.lookAway || Math.abs(yaw) > EYES.maxYaw;
    // A blink is not looking away; only count gaze while the eyes are open.
    if (this.closedSince === null) this.gaze.push({ t, on: !away });
    return this.state(t, this.closedSince !== null, away);
  }

  private state(t: number, closed: boolean, away: boolean): EyeState {
    const span = Math.min(EYES.windowSeconds, t - (this.start ?? t));
    const blinkRate = span >= 15 ? (this.blinks.length * 60) / span : null;
    const onScreen =
      this.gaze.length >= 30 ? this.gaze.filter((g) => g.on).length / this.gaze.length : null;
    return { blinkRate, onScreen, closed, away };
  }

  reset(): void {
    this.closedSince = null;
    this.blinks = [];
    this.gaze = [];
    this.start = null;
  }
}

/**
 * Head-yaw proxy from face-mesh landmarks: where the nose tip (1) sits between
 * the outer eye corners (33, 263). 0 = frontal, ±1 = fully turned.
 */
export function yawProxy(lm: { x: number; y: number }[]): number {
  const l = lm[33].x;
  const r = lm[263].x;
  const n = lm[1].x;
  const mid = (l + r) / 2;
  const half = Math.abs(r - l) / 2 || 1e-6;
  return Math.max(-1, Math.min(1, (n - mid) / half));
}

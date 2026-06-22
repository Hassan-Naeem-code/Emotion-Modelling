/**
 * Derive a coarse, human-readable label from a continuous valence/arousal point
 * using Russell's circumplex quadrants. DISPLAY ONLY — this is a heuristic, not
 * a classifier output, and the UI marks it "(derived)". Returns "neutral" near
 * the origin instead of forcing a quadrant guess on a weak signal.
 */
import { READOUT } from "./config";

export function coarseLabel(valence: number, arousal: number): string {
  if (Math.hypot(valence, arousal) < READOUT.neutralRadius) return "neutral";
  if (valence >= 0 && arousal >= 0) return "excited / happy";
  if (valence < 0 && arousal >= 0) return "tense / angry";
  if (valence < 0 && arousal < 0) return "sad / bored";
  return "calm / content"; // valence >= 0, arousal < 0
}

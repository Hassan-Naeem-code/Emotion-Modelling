/**
 * Single audit surface for all user-facing privacy/compliance copy AND the
 * abstention thresholds. Keep ALL such strings and tunables here so they can be
 * reviewed in one place (a project requirement).
 */

export const PRIVACY = {
  consentTitle: "This analyzes YOUR camera on YOUR device. Nothing is uploaded.",
  consentBody:
    "This tool estimates your own valence (pleasant–unpleasant) and arousal " +
    "(calm–activated) from your front camera. All processing happens in this " +
    "browser. No frames or biometric signals ever leave your machine. Press " +
    "start when you're ready.",
  points: [
    "On-device. No frames or biometrics leave this browser.",
    "Estimates your own state at your request. Not a diagnostic tool.",
    "Not for assessing other people — employees, students, or candidates.",
    "It will say “not sure” rather than guess when the signal is weak.",
  ],
  fineprint:
    "Research demo. Not medical or psychological advice. Does not detect " +
    "“true feelings”, deception, or mental health conditions.",
  liveFineprint:
    "Estimates your own state. Not a diagnostic tool. Nothing is uploaded.",
};

/**
 * Abstention thresholds. These make the uncertainty UX REAL from v0, regardless
 * of model quality. model_meta.json (written by export/to_onnx.py) can override
 * `uncertaintyStd` with the conformal half-width so the browser uses the SAME
 * calibrated bound as the research pipeline.
 */
export const THRESHOLDS = {
  // Predictive std above which we abstain ("genuinely ambiguous").
  uncertaintyStd: 0.35,
  // Mean luma (0–255) below which the scene is too dark to call.
  minBrightness: 45,
  // Mean luma above which the scene is blown out.
  maxBrightness: 235,
  // Face bounding-box area as fraction of frame; below this the face is too small.
  minFaceArea: 0.03,
  // Inter-frame landmark motion (normalized) above which there's too much motion.
  maxMotion: 0.06,
  // Frames a face must be absent before we declare "no face".
  noFaceGrace: 5,
  // Smoothing factor for the displayed VA point (0 = none, 1 = frozen).
  smoothing: 0.6,
};

/**
 * Optional coarse categorical readout derived from the continuous VA estimate,
 * for DISPLAY ONLY. This is a Russell-circumplex heuristic, not a classifier and
 * not a ground-truth emotion — the UI labels it "(derived)". Per project rules,
 * discrete emotions are never the model's target; this is a convenience overlay.
 */
export const READOUT = {
  disclaimer: "derived from valence/arousal — a heuristic label, not a measurement",
  // Radius around the origin within which we call it neutral rather than guess.
  // A calm, non-smiling face at a webcam reads as mild negative valence + low
  // arousal; without a generous deadzone that always tips to "sad/bored", which
  // is misleadingly harsh for an ordinary neutral expression.
  neutralRadius: 0.35,
};

export const ABSTAIN_REASONS = {
  NO_FACE: "No face detected",
  LOW_LIGHT: "Too dark to call",
  BRIGHT: "Overexposed",
  SMALL_FACE: "Move closer — face too small",
  MOTION: "Too much motion",
  AMBIGUOUS: "Genuinely ambiguous expression",
  OCCLUDED: "Face partially occluded",
} as const;

export type AbstainReason =
  (typeof ABSTAIN_REASONS)[keyof typeof ABSTAIN_REASONS] | null;

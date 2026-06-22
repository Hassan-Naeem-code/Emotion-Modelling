# Demo testing

## Automated (logic) tests

The abstention logic is pure and unit-tested — no browser, camera, or model
needed:

```bash
cd demo
npm install
npm test          # vitest: src/abstain.test.ts
```

These cover every abstention trigger (no-face, low-light, overexposure,
small-face, motion, ambiguous) plus confidence/quality monotonicity.

## Manual checklist (requires a human + webcam)

The camera/landmark/render path can only be verified by a person. Run
`npm run dev` and walk through this list. ✅ each.

### Consent & privacy
- [ ] On load, **no camera light / permission prompt** until you press **Start**.
- [ ] Consent screen shows the on-device + "not for assessing others" copy.
- [ ] Pressing **Stop** turns the camera off (OS camera indicator goes dark).

### Normal operation (well-lit, face centered, still)
- [ ] A dot appears on the VA plane with a small halo.
- [ ] Details panel shows valence, arousal, confidence %, quality %, model name.
- [ ] Model name reads `SYNTHETIC placeholder` (or `ONNX (...)` if you exported).
- [ ] Moving your face changes the dot position smoothly (no jitter spikes).

### Abstention triggers (each should flip to "Not sure" with the right reason)
- [ ] **Cover the camera / leave frame** → "No face detected" after ~5 frames.
- [ ] **Dim the lights** (or cover lens partially) → "Too dark to call".
- [ ] **Shine a bright light / point at a window** → "Overexposed".
- [ ] **Step far back** so the face is tiny → "Move closer — face too small".
- [ ] **Shake your head quickly** → "Too much motion".
- [ ] With a real high-uncertainty model, a **neutral/ambiguous face** →
      "Genuinely ambiguous expression".

### Uncertainty visualization
- [ ] Halo is **small** when still and well-framed (confident).
- [ ] Halo grows / state flips toward the frame edges or in poor conditions.
- [ ] In any "Not sure" state the **dot is replaced** by the not-sure message and
      the reason is shown both on the plane and in the details panel.

### Privacy verification (do this once)
- [ ] Open DevTools → Network tab, run for a minute → **no outbound requests
      carrying image/frame data** (only static asset + the MediaPipe model/wasm
      fetch at startup; no frame uploads).

## Swapping in the trained model

After exporting (`python -m export.to_onnx ...` from the repo root) so that
`demo/public/model.onnx` exists, reload and confirm the model name changes to
`ONNX (evidential)` and the abstention threshold reflects the conformal bound.

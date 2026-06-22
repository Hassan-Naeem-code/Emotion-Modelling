# Track B — On-device affect demo

Webcam → MediaPipe face landmarks → aligned crop → ONNX model → valence/arousal
estimate **with a calibrated confidence band and an explicit abstention state**.
Everything runs in the browser. No frames or biometrics ever leave the machine.

## Run

```bash
cd demo
npm install
npm run dev      # open the printed http://localhost:5173
```

The camera does **not** start until you press **Start** on the consent screen.

## Model: it works at v0 without training

On a fresh checkout there is no `public/model.onnx`, so the app runs in
**SYNTHETIC mode** — a transparent placeholder predictor clearly labeled in the
details panel. The point of v0 is that the **uncertainty + abstention UX is
real regardless of model quality**: abstention on no-face / low-light /
overexposure / too-small-face / motion all work immediately.

To drop in the trained Track A model (no UI changes needed):

```bash
# from the repo root, after training (see top-level README)
python -m export.to_onnx \
  --config configs/evidential.yaml \
  --ckpt results/evidential/seed1337/best.pt \
  --out demo/public/model.onnx \
  --conformal results/evidential/metrics.json
```

This writes `public/model.onnx` + `public/model_meta.json`. Reload the page; the
details panel will now read `ONNX (evidential)` and the abstention threshold
comes from the conformal half-width — the **same calibrated bound as the paper**.

## What you see

- **VA plane:** a dot at the current valence/arousal, with a translucent halo
  whose size = predictive uncertainty (small = confident, big = unsure).
- **Not-sure state:** when uncertainty exceeds the conformal threshold, or the
  signal is unreliable, the dot is replaced by **“Not sure — signal too weak to
  call”** plus the reason.
- **Details panel (the honesty surface):** raw valence, arousal, confidence %,
  signal-quality %, abstention reason, and which model is active.

## Privacy / compliance

All user-facing copy and every threshold live in one auditable file:
[src/config.ts](src/config.ts). This tool estimates **your own** state at your
request. It is **not** a diagnostic tool and **not** for assessing other people.

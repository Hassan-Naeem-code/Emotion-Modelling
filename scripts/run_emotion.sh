#!/usr/bin/env bash
# Self-contained: train the emotion classifier, then export it into the demo.
# Completes on its own (detached-friendly).
#   nohup bash scripts/run_emotion.sh > results/emotion_run.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-10}"
echo "[$(date +%H:%M:%S)] EMOTION TRAIN start (${EPOCHS} epochs)"
python3 -m train.train_emotion --epochs "$EPOCHS" --backbone efficientnet_b0

echo "[$(date +%H:%M:%S)] EXPORT emotion -> demo"
python3 -m export.emotion_to_onnx --ckpt results/emotion/best.pt --out demo/public/emotion.onnx

python3 - <<'PY'
import json
h = json.load(open("results/emotion/history.json"))
print(f"EMOTION_DONE val_acc={h['best_acc']:.3f} classes={','.join(h['classes'])}")
PY
echo "[$(date +%H:%M:%S)] ALL DONE"

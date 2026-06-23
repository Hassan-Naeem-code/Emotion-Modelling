#!/usr/bin/env bash
# Self-contained real-AffectNet pipeline: train -> eval -> export to demo ->
# aggregate. Designed to run fully detached and complete on its own, so the
# demo + results update even if no agent is watching.
#
#   setsid nohup bash scripts/run_real_affectnet.sh > results/real_run.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT=results/affectnet_va_evidential/seed1337/best.pt
EPOCHS="${EPOCHS:-15}"

echo "[$(date +%H:%M:%S)] TRAIN start (${EPOCHS} epochs)"
python3 -m train.train --config configs/affectnet_va.yaml \
  --set model.backbone=efficientnet_b0 --set train.epochs="$EPOCHS" \
  --set data.num_workers=2 --set data.batch_size=32

echo "[$(date +%H:%M:%S)] EVAL"
python3 -m eval.run_eval --config configs/affectnet_va.yaml --ckpt "$CKPT" \
  --set model.backbone=efficientnet_b0

echo "[$(date +%H:%M:%S)] EXPORT real model -> demo"
python3 -m export.to_onnx --config configs/affectnet_va.yaml --ckpt "$CKPT" \
  --out demo/public/model.onnx --conformal results/affectnet_va_evidential/metrics.json

echo "[$(date +%H:%M:%S)] AGGREGATE"
python3 -m eval.aggregate || true

# Final one-line summary marker the agent can grep for.
python3 - <<'PY'
import json
d = json.load(open("results/affectnet_va_evidential/metrics.json"))
a, c = d["acc"], d["conformal"]
print(f"REAL_DONE CCC_val={a['ccc_valence']:.3f} CCC_aro={a['ccc_arousal']:.3f} "
      f"RMSE={a['rmse_mean']:.3f} ECE={d['ece']:.3f} coverage={c['coverage_test']:.3f}")
PY
echo "[$(date +%H:%M:%S)] ALL DONE"

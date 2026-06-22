#!/usr/bin/env bash
# Train K deep-ensemble members, each with a different seed, into
# results/ensemble/memberN/. Epistemic uncertainty at eval = member disagreement.
#
#   bash scripts/train_ensemble.sh [K] [extra args passed to train.py]
#   bash scripts/train_ensemble.sh 5
#   bash scripts/train_ensemble.sh 3 --dry-run        # fast smoke test
set -euo pipefail

K="${1:-5}"
shift || true
EXTRA=("$@")

for s in $(seq 1 "$K"); do
  echo "=== ensemble member $s/$K (seed $s) ==="
  python -m train.train --config configs/ensemble_member.yaml \
    --seed "$s" --out "results/ensemble/member${s}" "${EXTRA[@]}"
done

echo "Trained $K members -> results/ensemble/. Evaluate with:"
echo "  python -m eval.run_eval --config configs/ensemble_eval.yaml \\"
echo "      --ckpt results/ensemble/member1/best.pt"

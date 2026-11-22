#!/usr/bin/env bash
# Leave-one-expression-out runner: for each seen expression k, train with k
# removed from Train and Validation, then dump predictions on the full Test set.
# Resumable like run_v2.sh: finished (method, held, seed) triples are skipped.
#
#   bash scripts/run_loeo.sh                                    # defaults below
#   HELD="0 1 2" SEEDS="1" bash scripts/run_loeo.sh             # split across machines
#   OUT=/content/drive/MyDrive/emotion_loeo bash scripts/run_loeo.sh
#
# Expressions: 0 neutral, 1 happy, 2 sad, 3 surprise, 4 fear, 5 disgust, 6 anger
# (7 contempt is already unseen in every run).
#
# Then:  python -m eval.loeo --preds $OUT/preds --out results/summary/loeo --ref results/v2/preds
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
METHODS=${METHODS:-"evidential"}
HELD=${HELD:-"0 1 2 3 4 5 6"}
SEEDS=${SEEDS:-"1 2 3"}
OUT=${OUT:-results/loeo}
MC_SAMPLES=${MC_SAMPLES:-20}
mkdir -p "$OUT/preds" "$OUT/ckpt"

for seed in $SEEDS; do
  for k in $HELD; do
    for m in $METHODS; do
      tag="loeo_${m}_h${k}_s${seed}"
      pred="$OUT/preds/${tag}.npz"
      ckdir="$OUT/ckpt/$tag"
      if [[ -f "$pred" ]]; then echo "skip $tag (done)"; continue; fi
      mkdir -p "$ckdir"
      if [[ ! -f "$ckdir/DONE" ]]; then
        echo "=== train $tag ($(date +%H:%M))"
        $PY -m train.train --config "configs/v2_${m}.yaml" --seed "$seed" --out "$ckdir" \
          --set data.holdout_expression="$k" ${TRAIN_ARGS:-} \
          2>&1 | grep --line-buffered -v -i warning | tee "$ckdir.log"
        touch "$ckdir/DONE"
      fi
      echo "=== predict $tag"
      # The held-out expression is read back from the checkpoint's config.
      $PY -m eval.dump_preds --ckpt "$ckdir/best.pt" --out "$pred"
      if [[ "$m" == "evidential" ]]; then
        $PY -m eval.dump_preds --ckpt "$ckdir/best.pt" --mc-samples "$MC_SAMPLES" \
          --out "$OUT/preds/loeo_mcdropout_h${k}_s${seed}.npz"
      fi
    done
  done
done
echo "all done -> $OUT/preds"

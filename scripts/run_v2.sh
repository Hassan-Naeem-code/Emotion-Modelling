#!/usr/bin/env bash
# v2 experiment runner: train -> dump predictions, for each method x seed.
# Resumable: a (method, seed) whose predictions already exist is skipped, so a
# dropped Colab session or a closed laptop just means "run it again".
#
#   bash scripts/run_v2.sh                       # everything, default seeds
#   METHODS="evidential" SEEDS="1 2" bash scripts/run_v2.sh
#   OUT=/content/drive/MyDrive/emotion_v2 bash scripts/run_v2.sh   # persist on Drive
#   PREFIX=v2c SEEDS="1 2 3" bash scripts/run_v2.sh  # ConvNeXt backbone check
#
# Then:  python -m eval.analyze --preds $OUT/preds --out results/summary/v2
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
METHODS=${METHODS:-"evidential gaussian_nll baseline"}
SEEDS=${SEEDS:-"1 2 3 4 5"}
OUT=${OUT:-results/v2}
MC_SAMPLES=${MC_SAMPLES:-20}
# Config/tag prefix: v2 (EfficientNet-B0) or v2c (ConvNeXt-Tiny backbone check).
PREFIX=${PREFIX:-v2}
mkdir -p "$OUT/preds" "$OUT/ckpt"

for seed in $SEEDS; do
  for m in $METHODS; do
    tag="${PREFIX}_${m}_s${seed}"
    pred="$OUT/preds/${tag}.npz"
    ckdir="$OUT/ckpt/$tag"
    if [[ -f "$pred" ]]; then echo "skip $tag (done)"; continue; fi
    if [[ ! -f "$ckdir/DONE" ]]; then
      echo "=== train $tag ($(date +%H:%M))"
      $PY -m train.train --config "configs/${PREFIX}_${m}.yaml" --seed "$seed" --out "$ckdir" ${TRAIN_ARGS:-} \
        2>&1 | grep --line-buffered -v -i warning | tee "$ckdir.log"
      touch "$ckdir/DONE"
    fi
    echo "=== predict $tag"
    $PY -m eval.dump_preds --ckpt "$ckdir/best.pt" --out "$pred" --corrupt
    # MC-dropout from the same evidential weights: a separate method row at no
    # training cost. (Evaluated at the training dropout rate, 0.2.)
    if [[ "$m" == "evidential" ]]; then
      $PY -m eval.dump_preds --ckpt "$ckdir/best.pt" --mc-samples "$MC_SAMPLES" \
        --out "$OUT/preds/${PREFIX}_mcdropout_s${seed}.npz" --corrupt
    fi
  done
done
echo "all done -> $OUT/preds"

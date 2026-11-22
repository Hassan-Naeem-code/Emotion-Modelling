# Leave-one-expression-out runs on Colab

Answers the main review risk: the unseen-expression result rests on one
expression (contempt, 70 test images). Here each of the seven seen expressions
is held out in turn: removed from Train and Validation, kept in Test.

| Held out | 0 neutral | 1 happy | 2 sad | 3 surprise | 4 fear | 5 disgust | 6 anger |
|---|---|---|---|---|---|---|---|
| Test images | 1264 | 2354 | 462 | 246 | 97 | 59 | 448 |

That is about 4,900 held-out test images in total, against 70 for contempt.

Default: evidential head (plus MC-dropout from the same weights, free),
3 seeds, so 21 trainings, roughly 40–60 min each on a T4. Split the work across
sessions or machines with `HELD` and `SEEDS`; finished runs are skipped.

**Runtime → Change runtime type → T4 (or better) GPU.**

## Cell 1 — code + deps

```python
!nvidia-smi -L
!git clone https://github.com/Hassan-Naeem-code/Emotion-Modelling.git
%cd Emotion-Modelling
!pip -q install timm pyyaml
```

## Cell 2 — data to local disk

```python
from google.colab import drive; drive.mount('/content/drive')
!mkdir -p data/raw && unzip -q -o /content/drive/MyDrive/AffectNetVA.zip -d data/raw/
!ls data/raw/AffectNetVA/Train/images | wc -l    # expect 58803
```

## Cell 3 — run (resumable)

```python
!mkdir -p /content/drive/MyDrive/emotion_loeo
!OUT=/content/drive/MyDrive/emotion_loeo SEEDS="1" HELD="0 1 2 3 4 5 6" \
  bash scripts/run_loeo.sh
```

Do seed 1 for all seven expressions first: that alone is enough to see whether
contempt's pattern repeats. Then seeds 2 and 3.

## Then (on the Mac, no GPU)

Copy `/content/drive/MyDrive/emotion_loeo/preds/*.npz` to `results/loeo/preds/`:

```bash
python -m eval.loeo --preds results/loeo/preds --out results/summary/loeo \
    --ref results/v2/preds
```

`results/summary/loeo/loeo_evidential.tex` is the paper table: for each held-out
expression, its coverage when it was in training, its coverage once held out
(per-axis and max-score), coverage on the other expressions, and the AUROC
of the uncertainty for flagging it.

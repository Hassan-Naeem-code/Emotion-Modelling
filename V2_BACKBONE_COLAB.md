# Backbone check on Colab (ConvNeXt-Tiny)

Answers the reviewer question "is this specific to EfficientNet-B0?". Same v2
recipe, protocol and analysis; only the backbone changes, to ConvNeXt-Tiny
(28M parameters, ImageNet-22k pretrained, peak LR 1e-4). Configs:
`configs/v2c_{evidential,gaussian_nll,baseline}.yaml`.

Plan: 3 heads × 3 seeds = 9 models (plus MC-dropout from each evidential model
for free). Rough cost per model: ~25 min on an A100, ~45 min on an L4, ~1.5–2 h
on a T4. Start with seed 1 only (3 models) if GPU time is short; that already
answers the question, the extra seeds give error bars.

**Runtime → Change runtime type → A100 or L4 if available, else T4.**

## Cell 1 — code + deps

Do NOT `pip install -r requirements.txt` (it pins numpy<2 and breaks Colab).

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

## Cell 3 — run (resumable; re-run after a disconnect and it continues)

```python
!mkdir -p /content/drive/MyDrive/emotion_v2c
!PREFIX=v2c OUT=/content/drive/MyDrive/emotion_v2c SEEDS="1 2 3" \
  METHODS="evidential gaussian_nll baseline" bash scripts/run_v2.sh
```

## Then (on the Mac)

Copy `/content/drive/MyDrive/emotion_v2c/preds/*.npz` into
`results/v2c/preds/` and run:

```bash
python -m eval.analyze --preds results/v2c/preds --out results/summary/v2c \
    --ensemble-of gaussian_nll,evidential
```

What to look for: per-axis joint coverage below 0.90 and the max-score region
at or above it (the diagnosis and fix), the worst seen expression, and
contempt coverage / AUROC (whether a stronger backbone notices the unseen
expression).

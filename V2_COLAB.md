# v2 runs on Colab (resubmission experiments)

Runs `scripts/run_v2.sh` on a Colab GPU, writing straight to Google Drive so a
dropped session loses nothing: re-run the last cell and it resumes where it
stopped (finished (method, seed) pairs are skipped).

Split the work with the Mac by giving each machine different seeds, e.g.
Colab `SEEDS="4 5"` and Mac `SEEDS="1 2 3"`. Results merge by copying the
`.npz` files into one folder.

**Runtime → Change runtime type → T4 (or better) GPU.**

## Cell 1 — code + deps

Do NOT `pip install -r requirements.txt` here: it pins numpy<2, which breaks
Colab's numpy-2 kernel ("numpy.dtype size changed"). Colab already has torch,
torchvision, scipy and pandas; only timm is missing.

```python
!nvidia-smi -L
!git clone https://github.com/Hassan-Naeem-code/Emotion-Modelling.git
%cd Emotion-Modelling
!pip -q install timm pyyaml
```

## Cell 2 — data to local disk (reading JPEGs from Drive is far too slow)

```python
from google.colab import drive; drive.mount('/content/drive')
!mkdir -p data/raw && unzip -q -o /content/drive/MyDrive/AffectNetVA.zip -d data/raw/
!ls data/raw/AffectNetVA/Train/images | wc -l    # expect 58803
```

## Cell 3 — run (resumable)

```python
!mkdir -p /content/drive/MyDrive/emotion_v2
!OUT=/content/drive/MyDrive/emotion_v2 SEEDS="4 5" \
  METHODS="evidential gaussian_nll baseline" bash scripts/run_v2.sh
```

Each model takes roughly 40–60 min on a T4 (10 epochs). If the session drops,
reconnect, run Cells 1–2 again, then Cell 3: it resumes.

## Then

Copy `/content/drive/MyDrive/emotion_v2/preds/*.npz` next to the Mac's
`results/v2/preds/` and run:

```bash
python -m eval.analyze --preds results/v2/preds --out results/summary/v2
```

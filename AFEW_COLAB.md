# AFEW-VA second-dataset runs on Colab

Answers the second review risk: all main results are on AffectNet. Here the
same v2 recipe is trained, conformally calibrated and tested on AFEW-VA alone,
to check that (1) per-axis intervals fall short of joint coverage and (2) the
max-score region restores it, on a second, independent dataset (video, in the
wild, different annotators).

AFEW-VA is video, so the split is by **clip**: 60% Train / 20% Validation
(calibration) / 20% Test, with 10% of the Train clips held out for model
selection. Frames of one clip never appear in two splits. The split is fixed
(not tied to the seed). Coverage is reported per frame and per clip
(`cov_joint_clip`: each clip counts once), because frames in a clip are
correlated.

## 0 — get the data (once, by hand)

1. Request access at https://ibug.doc.ic.ac.uk/resources/afew-va-database/
   and accept the licence (research use; no redistribution).
2. Download all parts and unzip them so that you get one folder per clip
   (`001/`, `002/`, ...), each holding `<clip>.json` and its frame images.
3. Zip that so the top-level folder is `AFEW-VA/` and upload it to Drive as
   `MyDrive/AFEW-VA.zip`.

Check locally first if you like (no GPU):

```bash
python -c "from data.loaders.afew_va import _read_manifest as r; from pathlib import Path; \
rows=r(Path('data/raw/AFEW-VA')); print(len(rows),'frames', len({x[3] for x in rows}),'clips')"
```

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
!mkdir -p data/raw && unzip -q -o /content/drive/MyDrive/AFEW-VA.zip -d data/raw/
!ls data/raw/AFEW-VA | head -3; ls data/raw/AFEW-VA | wc -l    # expect ~600 clips
```

## Cell 3 — run (resumable)

```python
!mkdir -p /content/drive/MyDrive/emotion_afew
!OUT=/content/drive/MyDrive/emotion_afew PREFIX=v2_afew CORRUPT= SEEDS="1 2 3" \
  METHODS="evidential gaussian_nll baseline" bash scripts/run_v2.sh
```

`CORRUPT=` (empty) matters: the corruption split is built from AffectNet and
does not apply here. AFEW-VA is roughly half AffectNet's size, so each model
should be noticeably quicker than the AffectNet runs. Add seeds 4 and 5 later
to match the main table.

## Then (on the Mac, no GPU)

Copy `/content/drive/MyDrive/emotion_afew/preds/*.npz` to `results/afew/preds/`:

```bash
python -m eval.analyze --preds results/afew/preds --out results/summary/afew \
    --prefix v2_afew --tables results/summary/afew/tables
```

The numbers to compare with AffectNet: `cov_perdim_joint` (expected below 90%),
`cov_joint` and `cov_bonf` (both expected at or above 90%), `area_joint` vs
`area_bonf` (max-score expected smaller), and the same three coverages per clip.

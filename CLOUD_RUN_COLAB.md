# Run the real experiments on a free cloud GPU (Google Colab)

Local Mac training stalls (throttled pretrained-weight downloads + slow MPS). Colab
gives a free NVIDIA GPU that runs the whole sweep in a fraction of the time and doesn't
hit those issues. ~20 min of setup, then it runs on its own.

## One-time: put AffectNet on Google Drive
The code is on GitHub, but the dataset (`data/raw/AffectNetVA/`, ~1.3 GB) is not.
1. On your Mac, zip it: right-click `Emotion-Modeling/data/raw/AffectNetVA` → **Compress**.
2. Upload `AffectNetVA.zip` to your Google Drive (top level of *My Drive*).

## In Colab (https://colab.research.google.com → New notebook)
First: **Runtime → Change runtime type → T4 GPU → Save.** Then paste these cells and run in order.

**Cell 1 — confirm GPU**
```python
!nvidia-smi -L
```

**Cell 2 — get the code**
```python
!git clone https://github.com/Hassan-Naeem-code/Emotion-Modelling.git
%cd Emotion-Modelling
!pip -q install -r requirements.txt
```

**Cell 3 — bring in the data from Drive**
```python
from google.colab import drive; drive.mount('/content/drive')
!mkdir -p data/raw
!unzip -q /content/drive/MyDrive/AffectNetVA.zip -d data/raw/
# expect: data/raw/AffectNetVA/{Train,Validation,Test}
!ls data/raw/AffectNetVA
```

**Cell 4 — run all methods on real AffectNet** (a few GPU-hrs; leave it running)
```python
import subprocess
for cfg in ["baseline", "gaussian_nll", "evidential", "conformal"]:
    print("=== training", cfg, "===")
    subprocess.run(["python","-m","train.train","--config",f"configs/{cfg}.yaml",
        "--seed","1337","--set","data.name=affectnet_va",
        "--set","model.backbone=efficientnet_b0","--out",f"results/{cfg}_real"], check=True)
    subprocess.run(["python","-m","eval.run_eval","--config",f"configs/{cfg}.yaml",
        "--ckpt",f"results/{cfg}_real/best.pt","--set","data.name=affectnet_va"], check=True)
# MC-Dropout (reuses the evidential checkpoint)
subprocess.run(["python","-m","eval.run_eval","--config","configs/mc_dropout.yaml",
    "--ckpt","results/evidential_real/best.pt","--set","data.name=affectnet_va"], check=True)
```

**Cell 5 — build the comparison table + save results to Drive**
```python
!python -m eval.aggregate
!mkdir -p /content/drive/MyDrive/emotion_results
!cp -r results/*_real paper/comparison_table.* /content/drive/MyDrive/emotion_results/
print("Done — results copied to Drive/emotion_results")
```

## After it finishes
Download the `emotion_results` folder from Drive and drop the `results/*_real/` folders
back into your local repo — the paper's tables/figures then get refreshed from the real
numbers and the Results prose tightened to match.

## Notes
- Free Colab sessions can disconnect after a few hours; if it stops mid-sweep, re-run
  Cell 4 — completed methods are cached under `results/`.
- For AFEW-VA (the shift experiment): still requires dataset access
  (see `REPRODUCE_REAL.md`). The corruption-shift fallback there needs no extra data.

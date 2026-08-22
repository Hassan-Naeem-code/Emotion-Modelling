# Tier 1: three paper limitations removed with evaluation only

No retraining. Every run below reuses checkpoints already in
`Drive/MyDrive/emotion_results/`. Budget ~25 minutes on a T4.

What it produces:
1. **Conformal at 3 seeds** — removes the `‡ single seed` caveat from Table 1.
2. **MC-dropout at 3 seeds** — adds a table row, and tests whether the
   Limitations claim about MC-dropout is even true (see note below).
3. **Severity sweep 1–5** — turns the single shift number into a degradation
   curve, showing uncertainty tracks *how much* the input has drifted.

---

## Cell 1 — setup

```python
!git clone https://github.com/Hassan-Naeem-code/Emotion-Modelling.git
%cd Emotion-Modelling
!pip -q install -r requirements.txt

from google.colab import drive; drive.mount('/content/drive')
!mkdir -p data/raw
!unzip -q /content/drive/MyDrive/AffectNetVA.zip -d data/raw/
!ls data/raw/AffectNetVA && ls data/raw/AffectNetVA/Train/images | wc -l   # expect 58803
```

## Cell 2 — stage the three evidential checkpoints

```python
import os, shutil
SRC = "/content/drive/MyDrive/emotion_results"
SEEDS = {"1337": "evidential_real", "7": "evidential_s7", "21": "evidential_s21"}
for seed, folder in SEEDS.items():
    dst = f"results/evid_seed{seed}"
    os.makedirs(dst, exist_ok=True)
    shutil.copy(f"{SRC}/{folder}/best.pt", f"{dst}/best.pt")
    print(seed, os.path.getsize(f"{dst}/best.pt"), "bytes")
```

## Cell 3 — the runner

```python
import subprocess, os, json, shutil

OUT = "/content/drive/MyDrive/emotion_results/tier1"
os.makedirs(OUT, exist_ok=True)
ENV = {**os.environ, "PYTHONPATH": os.getcwd()}

def run(cfg, seed, tag, extra=()):
    """One eval. Returns the metrics dict, and saves it to Drive under `tag`."""
    cmd = ["python", "-m", "eval.run_eval",
           "--config", f"configs/{cfg}.yaml",
           "--ckpt", f"results/evid_seed{seed}/best.pt",
           "--set", "data.name=affectnet_va",
           "--set", "model.backbone=efficientnet_b0",
           "--ood-dataset", "affectnet_va_corrupt", *extra]
    r = subprocess.run(cmd, env=ENV, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAILED {tag}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
        return None
    # run_eval writes to results/<config name>/; copy it out before it is overwritten.
    src = f"results/{cfg}/metrics.json"
    m = json.load(open(src))
    shutil.copy(src, f"{OUT}/{tag}.json")
    print(f"{tag:28s} CCC {m['acc']['ccc_mean']:.4f}  ECE {m['ece']:.4f}  "
          f"AURC {m['aurc']:.4f}  OOD-AUROC {m['shift']['ood_auroc']:.4f}")
    return m
```

## Cell 4 — conformal at three seeds

```python
for seed in ["1337", "7", "21"]:
    run("conformal", seed, f"conformal_seed{seed}")
```

## Cell 5 — MC-dropout at three seeds

```python
# Evaluate at dropout 0.2 to MATCH what the checkpoints were trained with
# (_base.yaml uses 0.2; mc_dropout.yaml's own default of 0.3 does not match).
for seed in ["1337", "7", "21"]:
    run("mc_dropout", seed, f"mcdropout_seed{seed}", extra=("--set", "model.dropout=0.2"))
```

If this errors, the paper's Limitations claim stands and nothing is lost — paste
me the error. If it runs, we gain a table row and drop the caveat.

## Cell 6 — severity sweep (the important one)

```python
for seed in ["1337", "7", "21"]:
    for sev in [1, 2, 3, 4, 5]:
        run("evidential", seed, f"sweep_s{seed}_sev{sev}",
            extra=("--set", f"data.severity={sev}"))
```

Severity 3 should reproduce the numbers already in the paper — a useful
consistency check that the sweep is wired correctly.

## Cell 7 — confirm what landed

```python
import os
files = sorted(os.listdir(OUT))
print(len(files), "result files in Drive/emotion_results/tier1:")
for f in files: print(" ", f)
```

Expect 21 files: 3 conformal + 3 mc-dropout + 15 sweep.

---

## Then

Tell me it's done and point me at `emotion_results/tier1/`. I will pull the
files down, commit them under `results/summary/real/`, extend
`eval/aggregate_seeds.py` to report conformal and MC-dropout with error bars,
and add the severity-degradation figure to the paper.

**Do not** overwrite the existing committed metrics with these; they are
additional runs, not replacements. The published v1 numbers stay as they are.

# Getting to paper-grade results (GPU + AFEW-VA)

The code is done. What's missing is **execution on real data**. This guide covers
(1) where to get a GPU, (2) the exact commands to run all methods on real
AffectNet, and (3) how to get AFEW-VA for the shift experiment — plus fallbacks
if you can't.

## Where you stand today
- ✅ Real AffectNet-VA data is already on disk (`data/raw/AffectNetVA/`).
- ✅ One real run done: `results/affectnet_va_evidential/` (in-distribution only).
- ❌ The 6-method comparison in the paper is still **synthetic** — must re-run on real data.
- ❌ `mc_dropout` and `ensemble` never run at all.
- ❌ The AffectNet→AFEW-VA shift never run (AFEW-VA not downloaded).

---

## 1. Get a GPU (ranked cheapest-effort first)

| Option | Cost | Notes |
|---|---|---|
| **Kaggle Notebooks** | Free | 30 GPU-hrs/week (P100/T4). AffectNet is uploadable as a private dataset. Best free option for a full sweep. |
| **Google Colab** | Free / ~$10-12/mo Pro | Free T4 with ~12h session cap; Pro gives longer runtimes + better GPUs. Put `data/raw/` on Google Drive and mount it. |
| **Vast.ai / RunPod / Lambda** | ~$0.20–0.50/hr | Rent a consumer GPU (RTX 3090/4090). Cheapest for the *entire* sweep in one sitting (~a few dollars total). `git clone`, `pip install -r requirements.txt`, run. |
| **Your Mac (Apple Silicon, MPS)** | Free | PyTorch runs on `mps`. EfficientNet-B0 will train but slowly — fine for 1–2 methods, painful for all 6 + ensembles. |

**Rough budget:** all 6 methods on AffectNet ≈ a few GPU-hours total on a 3090.
On a cloud rental that's roughly **$2–5**. On Kaggle it's free.

## 2. Run all methods on REAL AffectNet

Override the synthetic default (`configs/_base.yaml` → `data.name: synthetic`)
with `--set data.name=affectnet_va`. From the repo root, after
`pip install -r requirements.txt`:

```bash
# Point-estimate baseline, heteroscedastic Gaussian, evidential (primary), conformal
for cfg in baseline gaussian_nll evidential conformal; do
  python -m train.train --config configs/$cfg.yaml --seed 1337 \
    --set data.name=affectnet_va --out results/${cfg}_real
  python -m eval.run_eval --config configs/$cfg.yaml \
    --ckpt results/${cfg}_real/best.pt --set data.name=affectnet_va
done

# MC-Dropout (uses the evidential checkpoint, N stochastic passes)
python -m eval.run_eval --config configs/mc_dropout.yaml \
  --ckpt results/evidential_real/best.pt --set data.name=affectnet_va

# Deep ensemble: 5 seeds, then eval with the ensemble wrapper
for s in 1 2 3 4 5; do
  python -m train.train --config configs/ensemble_member.yaml --seed $s \
    --set data.name=affectnet_va --out results/ensemble/member$s
done
python -m eval.run_eval --config configs/ensemble_eval.yaml \
  --set data.name=affectnet_va uncertainty.ensemble_dir=results/ensemble

# Regenerate the paper's table + figures from the REAL runs
make aggregate
```

**Multi-seed error bars (for a credible paper):** repeat the first loop with
`--seed 1 2 3` and average — reviewers expect variance, not a single seed.

After this, replace the synthetic figures the paper embeds (`evidential_*.png`)
with the real-run versions so Results reports real numbers.

## 3. AFEW-VA for the shift experiment

**Request access** (academic, EULA-gated):
- Official page: **https://ibug.doc.ic.ac.uk/resources/afew-va-database/**
- Fill the request form; turnaround is typically days. Extract into
  `data/raw/AFEW-VA/` (the loader `data/loaders/afew_va.py` expects it there).

Then run the real shift:
```bash
python -m eval.run_eval --config configs/shift_affectnet_to_afew.yaml \
  --ckpt results/evidential_real/best.pt \
  --set data.name=affectnet_va --ood-dataset afew_va
```

**If you can't get AFEW-VA**, use an achievable shift instead and reword the paper
from "cross-dataset" to "corruption/robustness shift":
- **Corruption shift (recommended):** apply common corruptions (Gaussian blur,
  noise, brightness, JPEG) to the AffectNet test set and measure calibration
  degradation + OOD-AUROC — the ImageNet-C style protocol. This needs a small
  loader that wraps the existing test set; ask and I'll add it.
- **Held-out split shift:** carve AffectNet by an attribute and treat one side as OOD.

Either fallback keeps the "flags its own degradation under shift" claim honest
with data you actually have.

---

## What to hand back to me after a run
Point me at the new `results/*_real/metrics.json` files (or just say "done") and
I'll: refresh the paper's table/figures from real numbers, tighten the Results
prose to the actual findings, and restore the strong shift claim if the real
shift supports it.

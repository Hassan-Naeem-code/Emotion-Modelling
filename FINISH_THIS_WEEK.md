# Finishing the paper — what's done and what's left

## Status

The paper's headline table is now **generated from real runs**, not hand-written.
The backing artifacts are committed under `results/summary/real/<method>/seed<n>/metrics.json`
(recovered from `Drive/MyDrive/emotion_results/`), and

```bash
make paper-table      # -> paper/comparison_table.{tex,csv}
```

reproduces the exact table in the paper from those files. If a number in the paper
ever disagrees with a run, this command is the tie-breaker.

Seeds recovered: **1337** (the `*_real` folders), **7** (`*_s7`), **21** (`*_s21`),
for baseline / gaussian_nll / evidential. Conformal is seed 1337 only — hence the
`‡` in the table.

## The one remaining gap: Figures 1a–1c are still synthetic

`paper/evidential_{risk_coverage,reliability,va_scatter}.png` are dated **Jun 22**
and come from the *synthetic* smoke-test run (the one with CCC 0.97), not from the
real AffectNet runs the table reports. **These must be regenerated before submission** —
a real table with synthetic figures is the kind of thing a reviewer catches.

The fix is cheap: the trained checkpoints are already on your Drive, so this is
evaluation only (~2 minutes on a T4), no retraining.

### Colab cell — regenerate the real figures

Runtime → Change runtime type → T4 GPU. Then Cells 1–3 from `CLOUD_RUN_COLAB.md`
(clone + install + mount Drive and unzip the data), then:

```python
import subprocess, shutil, os
# Use the seed-1337 evidential checkpoint that produced the reported numbers.
os.makedirs("results/evidential_real", exist_ok=True)
shutil.copy("/content/drive/MyDrive/emotion_results/evidential_real/best.pt",
            "results/evidential_real/best.pt")

subprocess.run(["python","-m","eval.run_eval","--config","configs/evidential.yaml",
    "--ckpt","results/evidential_real/best.pt",
    "--set","data.name=affectnet_va","--set","model.backbone=efficientnet_b0",
    "--ood-dataset","affectnet_va_corrupt"], check=True)

# The three figures the paper embeds, plus the metrics they correspond to.
dst = "/content/drive/MyDrive/emotion_results/figures_real"
os.makedirs(dst, exist_ok=True)
for f in ["evidential_risk_coverage.png","evidential_reliability.png",
          "evidential_va_scatter.png"]:
    shutil.copy(f"paper/{f}", f"{dst}/{f}")
shutil.copy("results/evidential/metrics.json", f"{dst}/metrics.json")
print("figures ->", dst)
```

**Sanity check before you use them:** the printed `metrics.json` should match the
committed `results/summary/real/evidential/seed1337/metrics.json` —
CCC 0.6826, ECE 0.01508, AURC 0.04721, OOD-AUROC 0.60646. If it does, the figures
are from the same run as the table and you're clean. If it doesn't, tell me the
numbers before using the figures.

Then download `figures_real/` and drop the three PNGs into `paper/`, overwriting
the synthetic ones.

### Same trip: give the demo a real model

While that Colab session is open, the browser demo still ships a synthetic
placeholder:

```python
subprocess.run(["python","-m","export.to_onnx","--config","configs/evidential.yaml",
    "--ckpt","results/evidential_real/best.pt",
    "--set","model.backbone=efficientnet_b0"], check=True)
shutil.copy("demo/public/model.onnx", "/content/drive/MyDrive/emotion_results/model.onnx")
shutil.copy("demo/public/model_meta.json", "/content/drive/MyDrive/emotion_results/model_meta.json")
```

Download both into `demo/public/`. The demo picks them up with no code change.

## Confirm one thing

The Colab runs used `configs/_base.yaml` defaults except for the backbone
override (`efficientnet_b0`). The 4-worker DataLoader warning in every eval log
confirms `num_workers: 4`, i.e. the base defaults, so `experiments.tex` states
**30 epochs, batch size 64, AdamW lr 3e-4, wd 1e-4, grad clip 1.0**. If you
changed epochs or batch size in the notebook, tell me and I'll correct the paper —
this is the one number in the setup I inferred rather than read off an artifact.

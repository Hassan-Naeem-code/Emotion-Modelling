# Calibrated, Uncertainty-Aware Multimodal Affect Estimation

Two deliverables, one model core:

- **Track A (research):** a reproducible pipeline for **calibrated,
  uncertainty-aware valence–arousal (VA) estimation under distribution shift** —
  produces the paper's tables and figures.
- **Track B (demo):** a browser app that runs the model on your **own** front
  camera, shows a VA estimate **with a confidence band**, and **abstains**
  ("not sure") when the signal is unreliable. **All inference is on-device.**

> The contribution is **knowing when it doesn't know**. Calibration and
> abstention are the point — not confident-looking outputs. Targets are
> continuous VA in [-1, 1], never the 7 discrete "basic emotions" (an optional
> coarse readout may be derived from VA *for display only*).

## Quickstart (Track A)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# End-to-end smoke test on synthetic data — no real dataset needed:
make dryrun
```

`make dryrun` trains the evidential model for 2 epochs on a tiny synthetic VA
dataset and runs the full evaluation, writing metrics + figures to `results/`.
This proves the whole machinery (training → uncertainty → conformal → metrics →
figures) end-to-end before you touch real data.

## Full reproduction

```bash
make repro       # train baseline+gaussian+evidential, eval all, emit tables/figures
```

Everything is config-driven and seeded — every number traces to a
`configs/*.yaml` + a `--seed` + a file under `results/`. Edit `configs/_base.yaml`
to point `data.name` at a real dataset once downloaded (see
[data/loaders/README.md](data/loaders/README.md) — nothing is auto-downloaded).

## Methods (all swappable via config)

| Config | Head | Uncertainty |
|---|---|---|
| `baseline.yaml` | regression (MSE) | none |
| `gaussian_nll.yaml` | heteroscedastic Gaussian | aleatoric |
| `evidential.yaml` | **Deep Evidential Regression** (primary) | aleatoric + epistemic |
| `mc_dropout.yaml` | evidential ckpt | epistemic via N stochastic passes |
| `ensemble_member.yaml` | gaussian | epistemic via deep ensemble |
| `conformal.yaml` | evidential ckpt | **split conformal** coverage guarantee |

## Metrics produced

- **Accuracy:** CCC (valence, arousal), RMSE, MAE.
- **Calibration:** regression ECE (quantile calibration), sharpness.
- **Selective prediction:** risk–coverage / accuracy–rejection curves, **AURC**.
- **Shift:** AffectNet→AFEW-VA (synthetic in-dist→OOD in dryrun) calibration
  degradation; **OOD AUROC** from the uncertainty score.
- **Conformal:** empirical coverage vs target on in-dist and shifted data.

Outputs land in `results/<name>/` and headline artifacts copy to `paper/`:
`metrics.json`, `results_table.{csv,tex}`, `risk_coverage.png`,
`reliability.png`, `va_scatter.png`.

## Deep Ensemble (epistemic)

```bash
for s in 1 2 3 4 5; do
  python -m train.train --config configs/ensemble_member.yaml --seed $s \
    --out results/ensemble/member$s
done
# then eval with uncertainty.wrapper=ensemble, ensemble_dir=results/ensemble
```

## Track B (demo)

```bash
cd demo && npm install && npm run dev
```

Runs immediately with a labeled **synthetic placeholder** model; the
abstention/uncertainty UX is real from v0. Drop in the trained model with
`python -m export.to_onnx ...` (see [demo/README.md](demo/README.md)) — no UI
changes. **No frames leave the browser.**

## Repository layout

```
configs/   YAML for every experiment (no magic numbers in code)
data/      loaders (synthetic + AffectNet/AFEW-VA/RAF-DB) + deterministic splits
models/    backbone, heads (regression/gaussian/evidential), rppg, fusion, wrappers
train/     train.py + losses.py
eval/      metrics.py, selective.py, shift.py, run_eval.py
export/    to_onnx.py (PyTorch -> ONNX for the demo)
demo/      Track B (Vite + TS, MediaPipe, ONNX Runtime Web)
results/   JSON/CSV/PNG (gitignored except results/summary/)
paper/     headline figures/tables
```

## Non-goals (intentionally not built)

No monitoring of other people (employees/students/candidates). No deception or
mental-health detection. No "true feelings" claims. No cloud upload of any frame
or biometric. No discrete-emotion classifier as the primary model.

## rPPG (modality two)

`models/rppg.py` extracts a pulse signal (POS algorithm, no training) and a
**signal-quality score that gates its contribution** — low quality down-weights
the rPPG feature and *raises* uncertainty rather than guessing. Full video-window
fusion is scaffolded (`models/fusion.py`, `model.use_rppg`) for later.

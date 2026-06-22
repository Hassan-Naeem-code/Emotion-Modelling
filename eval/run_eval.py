"""Produces all paper tables and figures for one experiment.

    python -m eval.run_eval --config configs/evidential.yaml \
        --ckpt results/evidential/seed1337/best.pt [--dry-run]

Emits into results/<name>/ and copies the headline artifacts into paper/:
  - metrics.json                 full metric dump
  - results_table.csv / .tex     LaTeX-ready results table
  - risk_coverage.png            selective-prediction payoff
  - reliability.png              calibration reliability diagram
  - va_scatter.png               VA predictions colored by uncertainty
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from common import (apply_overrides, ensure_dir, get_device,  # noqa: E402
                    load_config, set_seed)
from data.build import build_datasets  # noqa: E402
from eval.metrics import (accuracy_metrics, empirical_coverage,  # noqa: E402
                          regression_ece, sharpness, uncertainty_score)
from eval.selective import aurc, risk_coverage_curve  # noqa: E402
from eval.shift import shift_report  # noqa: E402
from models.fusion import AffectModel  # noqa: E402
from models.wrappers import (conformal_calibrate, conformal_intervals,  # noqa: E402
                             predict_ensemble, predict_mc_dropout, predict_plain)


def _loader(ds, cfg):
    return DataLoader(ds, batch_size=cfg["data"]["batch_size"], shuffle=False,
                      num_workers=cfg["data"]["num_workers"])


def _predict(cfg, model, ds, device):
    """Dispatch to the configured uncertainty wrapper."""
    loader = _loader(ds, cfg)
    wrapper = cfg["uncertainty"]["wrapper"]
    if wrapper == "mc_dropout":
        return predict_mc_dropout(model, loader, device, cfg["uncertainty"]["mc_samples"])
    if wrapper == "ensemble":
        return predict_ensemble(cfg, loader, device, cfg["uncertainty"]["ensemble_dir"])
    # 'none' and 'conformal' both start from a plain forward pass.
    return predict_plain(model, loader, device)


def _fig_risk_coverage(pred, path):
    cov, risk = risk_coverage_curve(pred["mean"], pred["var"], pred["target"])
    plt.figure(figsize=(4, 3))
    plt.plot(cov, risk, marker="o", ms=3)
    plt.xlabel("Coverage (fraction answered)")
    plt.ylabel("Risk (MSE on retained)")
    plt.title("Risk–Coverage")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _fig_reliability(pred, path):
    var = np.maximum(pred["var"], 1e-8)
    z = np.abs(pred["target"] - pred["mean"]) / np.sqrt(var)
    levels = np.linspace(0.05, 0.95, 19)
    emp = [np.mean(z <= stats.norm.ppf(0.5 + p / 2)) for p in levels]
    plt.figure(figsize=(4, 3))
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    plt.plot(levels, emp, marker="o", ms=3, label="model")
    plt.xlabel("Nominal coverage")
    plt.ylabel("Empirical coverage")
    plt.title("Reliability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _fig_va_scatter(pred, path):
    unc = uncertainty_score(pred["var"])
    plt.figure(figsize=(4, 4))
    sc = plt.scatter(pred["mean"][:, 0], pred["mean"][:, 1], c=unc, cmap="viridis",
                     s=10, alpha=0.7)
    plt.colorbar(sc, label="uncertainty")
    plt.xlabel("Valence")
    plt.ylabel("Arousal")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axhline(0, color="gray", lw=0.5)
    plt.axvline(0, color="gray", lw=0.5)
    plt.title("VA estimates (color = uncertainty)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _write_table(metrics, csv_path, tex_path):
    rows = [
        ("CCC (valence)", metrics["acc"]["ccc_valence"]),
        ("CCC (arousal)", metrics["acc"]["ccc_arousal"]),
        ("RMSE (mean)", metrics["acc"]["rmse_mean"]),
        ("Regression ECE", metrics["ece"]),
        ("Sharpness", metrics["sharpness"]),
        ("AURC", metrics["aurc"]),
        ("OOD AUROC", metrics["shift"]["ood_auroc"]),
        ("Conformal coverage (test)", metrics["conformal"]["coverage_test"]),
        ("Conformal coverage (OOD)", metrics["conformal"]["coverage_ood"]),
        ("Target coverage", metrics["conformal"]["target"]),
    ]
    import csv
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in rows:
            w.writerow([k, f"{v:.4f}"])
    lines = [r"\begin{tabular}{lr}", r"\toprule",
             r"Metric & Value \\", r"\midrule"]
    for k, v in rows:
        lines.append(f"{k} & {v:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(tex_path).write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="Override any config key, e.g. --set data.name=affectnet")
    ap.add_argument("--ood-dataset", default=None,
                    help="Build the OOD/shift split from a DIFFERENT dataset "
                         "(e.g. afew_va) for cross-dataset shift evaluation.")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    if args.dry_run:
        # Weights come from the checkpoint anyway; skip pretrained download.
        cfg["model"]["pretrained"] = False
    set_seed(cfg["seed"])
    device = get_device()

    try:
        train_ds, val_ds, test_ds, ood_ds = build_datasets(cfg, dry_run=args.dry_run)
    except FileNotFoundError as e:
        sys.exit(f"\n[data not available] {e}\n"
                 f"Dataset '{cfg['data']['name']}' is registration-gated and not "
                 f"bundled. Download it (see data/loaders/README.md) and place it "
                 f"under '{cfg['data']['root']}/', or use --set data.name=synthetic.")
    if test_ds is None:
        sys.exit(f"\n[no test split] Dataset '{cfg['data']['name']}' provides no "
                 f"test split to evaluate.")

    model = AffectModel(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])

    # Cross-dataset shift: optionally pull the OOD split from another dataset
    # (e.g. train/test on AffectNet, shift-test on AFEW-VA).
    if args.ood_dataset:
        from copy import deepcopy
        ood_cfg = deepcopy(cfg)
        ood_cfg["data"]["name"] = args.ood_dataset
        try:
            _, _, ood_test, ood_split = build_datasets(ood_cfg, dry_run=args.dry_run)
        except FileNotFoundError as e:
            sys.exit(f"\n[shift data not available] {e}\n"
                     f"The shift set '{args.ood_dataset}' is registration-gated. "
                     f"Download it (see data/loaders/README.md) or run without "
                     f"--ood-dataset to use the in-config dataset's own OOD split.")
        ood_ds = ood_split if ood_split is not None else ood_test

    pred_test = _predict(cfg, model, test_ds, device)
    pred_ood = _predict(cfg, model, ood_ds, device) if ood_ds is not None else pred_test

    # ---- core metrics ----
    acc = accuracy_metrics(pred_test["mean"], pred_test["target"])
    ece = regression_ece(pred_test["mean"], pred_test["var"], pred_test["target"],
                         n_bins=cfg["eval"]["ece_bins"])
    shp = sharpness(pred_test["var"])
    au = aurc(pred_test["mean"], pred_test["var"], pred_test["target"],
              n_steps=cfg["eval"]["reject_steps"])
    shift = shift_report(pred_test, pred_ood)

    # ---- split conformal: calibrate on val (fallback test), apply to test+ood ----
    cal_ds = val_ds if val_ds is not None else test_ds
    cal_pred = _predict(cfg, model, cal_ds, device)
    alpha = cfg["uncertainty"]["conformal"]["alpha"]
    calib = conformal_calibrate(cal_pred, alpha)
    iv_test = conformal_intervals(pred_test, calib)
    iv_ood = conformal_intervals(pred_ood, calib)
    conformal = {
        "alpha": alpha,
        "target": cfg["eval"]["target_coverage"],
        "quantiles": calib["q"].tolist(),
        "coverage_test": empirical_coverage(iv_test["lower"], iv_test["upper"],
                                            pred_test["target"]),
        "coverage_ood": empirical_coverage(iv_ood["lower"], iv_ood["upper"],
                                           pred_ood["target"]),
        "mean_half_width_test": float(np.mean(iv_test["half_width"])),
    }

    metrics = {"config": cfg["name"], "acc": acc, "ece": ece, "sharpness": shp,
               "aurc": au, "shift": shift, "conformal": conformal}

    out_dir = ensure_dir(Path(cfg["logging"]["out_dir"]) / cfg["name"])
    paper_dir = ensure_dir("paper")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    _write_table(metrics, out_dir / "results_table.csv", out_dir / "results_table.tex")
    _fig_risk_coverage(pred_test, out_dir / "risk_coverage.png")
    _fig_reliability(pred_test, out_dir / "reliability.png")
    _fig_va_scatter(pred_test, out_dir / "va_scatter.png")
    # Headline copies for the paper.
    for fn in ["risk_coverage.png", "reliability.png", "va_scatter.png",
               "results_table.tex"]:
        (paper_dir / f"{cfg['name']}_{fn}").write_bytes((out_dir / fn).read_bytes())

    print(json.dumps(metrics, indent=2))
    print(f"\nWrote results -> {out_dir}  (figures + table copied to paper/)")


if __name__ == "__main__":
    main()

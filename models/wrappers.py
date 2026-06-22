"""Eval-time uncertainty wrappers: MC-Dropout, Deep Ensemble, Split Conformal.

Each predictor returns a unified dict of numpy arrays:
    mean        (N,2)  predictive mean for valence/arousal
    var         (N,2)  predictive variance (aleatoric + epistemic where available)
    target      (N,2)  ground truth (when available)
Conformal additionally produces per-dim interval half-widths with coverage
guarantees, which the demo uses as its abstention threshold.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .fusion import AffectModel


def _enable_dropout(model: torch.nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


@torch.no_grad()
def _forward_collect(model, loader, device):
    means, vars, targets = [], [], []
    for batch in loader:
        x, y = batch[0].to(device), batch[1]
        out = model(x)
        means.append(out["mean"].cpu().numpy())
        if "var" in out:
            vars.append(out["var"].cpu().numpy())
        else:
            vars.append(np.zeros_like(out["mean"].cpu().numpy()))
        targets.append(y.numpy())
    return np.concatenate(means), np.concatenate(vars), np.concatenate(targets)


def predict_plain(model, loader, device):
    model.eval()
    mean, var, target = _forward_collect(model, loader, device)
    return {"mean": mean, "var": var, "target": target}


@torch.no_grad()
def predict_mc_dropout(model, loader, device, n_samples: int):
    """Epistemic uncertainty from N stochastic passes with dropout active."""
    model.eval()
    _enable_dropout(model)
    all_means, all_alea, targets = [], [], []
    # Collect per-pass means and (if present) aleatoric variances.
    sample_means = [[] for _ in range(n_samples)]
    sample_alea = [[] for _ in range(n_samples)]
    targ = []
    for bi, batch in enumerate(loader):
        x, y = batch[0].to(device), batch[1]
        targ.append(y.numpy())
        for s in range(n_samples):
            out = model(x)
            sample_means[s].append(out["mean"].cpu().numpy())
            sample_alea[s].append(
                out["var"].cpu().numpy() if "var" in out
                else np.zeros_like(out["mean"].cpu().numpy())
            )
    target = np.concatenate(targ)
    # (S, N, 2)
    stk_mean = np.stack([np.concatenate(m) for m in sample_means])
    stk_alea = np.stack([np.concatenate(a) for a in sample_alea])
    mean = stk_mean.mean(axis=0)
    epistemic = stk_mean.var(axis=0)          # disagreement across passes
    aleatoric = stk_alea.mean(axis=0)
    return {"mean": mean, "var": aleatoric + epistemic,
            "epistemic": epistemic, "aleatoric": aleatoric, "target": target}


def predict_ensemble(cfg, loader, device, ensemble_dir: str):
    """Deep ensemble: average members, epistemic = variance of member means."""
    ckpts = sorted(Path(ensemble_dir).glob("**/best.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No best.pt checkpoints under {ensemble_dir}")
    member_means, member_vars, target = [], [], None
    for ck in ckpts:
        model = AffectModel(cfg).to(device)
        model.load_state_dict(torch.load(ck, map_location=device)["model"])
        out = predict_plain(model, loader, device)
        member_means.append(out["mean"])
        member_vars.append(out["var"])
        target = out["target"]
    stk_mean = np.stack(member_means)         # (K,N,2)
    mean = stk_mean.mean(axis=0)
    epistemic = stk_mean.var(axis=0)
    aleatoric = np.stack(member_vars).mean(axis=0)
    return {"mean": mean, "var": aleatoric + epistemic,
            "epistemic": epistemic, "aleatoric": aleatoric, "target": target}


# --------------------------- Split conformal ---------------------------------

def conformal_calibrate(cal_pred: dict, alpha: float) -> dict:
    """Split-conformal calibration for per-dim prediction intervals.

    Nonconformity score is the variance-normalized absolute residual, so the
    interval adapts to the model's own uncertainty. Returns per-dim quantiles
    q such that the interval is  mean ± q * sqrt(var).

    Finite-sample correction: use the ceil((n+1)(1-alpha))/n empirical quantile.
    """
    mean, var, y = cal_pred["mean"], cal_pred["var"], cal_pred["target"]
    scale = np.sqrt(var) + 1e-6
    scores = np.abs(y - mean) / scale                # (N,2)
    n = scores.shape[0]
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    q = np.quantile(scores, level, axis=0)           # (2,)
    return {"q": q, "alpha": alpha, "level": float(level)}


def conformal_intervals(pred: dict, calib: dict) -> dict:
    """Apply calibrated quantiles to produce intervals + half-widths."""
    scale = np.sqrt(pred["var"]) + 1e-6
    half = calib["q"] * scale                         # (N,2)
    lower = pred["mean"] - half
    upper = pred["mean"] + half
    return {**pred, "lower": lower, "upper": upper, "half_width": half}

"""Metrics for calibrated VA estimation. All functions take numpy arrays shaped
(N,2) for valence/arousal unless noted, and return plain floats / small dicts so
results serialize cleanly to JSON/CSV.

Accuracy : CCC (the standard VA metric), RMSE, MAE.
Calibration: regression ECE (quantile/interval calibration), sharpness.
Selective : see eval/selective.py.
OOD      : AUROC separating in-dist vs shifted using the uncertainty score.
Conformal: empirical coverage vs target.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def ccc(pred: np.ndarray, target: np.ndarray) -> float:
    """Concordance Correlation Coefficient (Lin's CCC) for one dimension."""
    pred, target = np.asarray(pred), np.asarray(target)
    mp, mt = pred.mean(), target.mean()
    vp, vt = pred.var(), target.var()
    cov = ((pred - mp) * (target - mt)).mean()
    denom = vp + vt + (mp - mt) ** 2
    return float(2 * cov / denom) if denom > 0 else 0.0


def accuracy_metrics(mean: np.ndarray, target: np.ndarray) -> dict:
    dims = ["valence", "arousal"]
    out = {}
    for i, d in enumerate(dims):
        out[f"ccc_{d}"] = ccc(mean[:, i], target[:, i])
        out[f"rmse_{d}"] = float(np.sqrt(np.mean((mean[:, i] - target[:, i]) ** 2)))
        out[f"mae_{d}"] = float(np.mean(np.abs(mean[:, i] - target[:, i])))
    out["ccc_mean"] = 0.5 * (out["ccc_valence"] + out["ccc_arousal"])
    out["rmse_mean"] = 0.5 * (out["rmse_valence"] + out["rmse_arousal"])
    return out


def regression_ece(mean: np.ndarray, var: np.ndarray, target: np.ndarray,
                   n_bins: int = 15) -> float:
    """Calibration error for probabilistic regression via quantile calibration
    (Kuleshov et al., 2018). For a set of nominal coverage levels p, measure the
    empirical fraction of targets falling within the predicted central p-interval
    of a Gaussian(mean,var); ECE is the mean |empirical - nominal|.

    A perfectly calibrated model has empirical == nominal at every level.
    """
    var = np.maximum(var, 1e-8)
    std = np.sqrt(var)
    # z-scores of the residuals; for a calibrated Gaussian these are N(0,1).
    z = np.abs(target - mean) / std                  # (N,2)
    levels = np.linspace(0.05, 0.95, n_bins)
    err = 0.0
    for p in levels:
        # central interval half-width in z for coverage p.
        zc = stats.norm.ppf(0.5 + p / 2.0)
        empirical = np.mean(z <= zc)                 # over both dims and samples
        err += abs(empirical - p)
    return float(err / len(levels))


def sharpness(var: np.ndarray) -> float:
    """Average predicted std. Lower = sharper. Only meaningful alongside good
    calibration (sharp + calibrated is the goal; sharp + miscalibrated is bad)."""
    return float(np.mean(np.sqrt(np.maximum(var, 1e-8))))


def uncertainty_score(var: np.ndarray) -> np.ndarray:
    """Scalar per-sample uncertainty (total predictive std summed over dims)."""
    return np.sqrt(np.maximum(var, 1e-8)).sum(axis=1)


def ood_auroc(var_id: np.ndarray, var_ood: np.ndarray) -> float:
    """AUROC for separating in-dist vs OOD using the uncertainty score.
    Higher uncertainty should rank OOD above in-dist."""
    s_id = uncertainty_score(var_id)
    s_ood = uncertainty_score(var_ood)
    scores = np.concatenate([s_id, s_ood])
    labels = np.concatenate([np.zeros(len(s_id)), np.ones(len(s_ood))])
    # AUROC via Mann-Whitney U statistic (rank-based, no sklearn dependency).
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def empirical_coverage(lower: np.ndarray, upper: np.ndarray,
                       target: np.ndarray) -> float:
    """Fraction of targets inside [lower, upper], averaged over dims."""
    inside = (target >= lower) & (target <= upper)
    return float(inside.mean())

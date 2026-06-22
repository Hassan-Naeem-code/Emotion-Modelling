"""Selective prediction: risk–coverage and accuracy–rejection curves + AURC.

The model abstains on its most-uncertain inputs. We sort by the uncertainty
score, and for each coverage level (fraction of inputs we actually answer)
compute the risk (error) on the retained set. A good uncertainty estimate makes
risk drop sharply as we reject — that drop, summarized by AURC, is the whole
payoff of knowing when you don't know.
"""
from __future__ import annotations

import numpy as np

from .metrics import uncertainty_score


def risk_coverage_curve(mean: np.ndarray, var: np.ndarray, target: np.ndarray,
                        n_steps: int = 20):
    """Return (coverages, risks). Risk = MSE over the retained (most-confident)
    fraction at each coverage level."""
    unc = uncertainty_score(var)
    order = np.argsort(unc)                     # most confident first
    sq_err = ((mean - target) ** 2).mean(axis=1)[order]
    n = len(unc)
    coverages = np.linspace(1.0 / n_steps, 1.0, n_steps)
    risks = []
    for c in coverages:
        k = max(1, int(round(c * n)))
        risks.append(float(sq_err[:k].mean()))
    return coverages, np.array(risks)


def aurc(mean: np.ndarray, var: np.ndarray, target: np.ndarray,
         n_steps: int = 20) -> float:
    """Area under the risk–coverage curve (lower is better)."""
    cov, risk = risk_coverage_curve(mean, var, target, n_steps)
    # np.trapz was removed in numpy 2.x in favour of np.trapezoid.
    trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trap(risk, cov))


def accuracy_rejection_curve(mean: np.ndarray, var: np.ndarray,
                             target: np.ndarray, n_steps: int = 20):
    """Return (rejection_fractions, mae) as we reject the most-uncertain inputs."""
    unc = uncertainty_score(var)
    order = np.argsort(unc)
    abs_err = np.abs(mean - target).mean(axis=1)[order]
    n = len(unc)
    rejects = np.linspace(0.0, 1.0 - 1.0 / n_steps, n_steps)
    maes = []
    for r in rejects:
        k = max(1, int(round((1.0 - r) * n)))
        maes.append(float(abs_err[:k].mean()))
    return rejects, np.array(maes)

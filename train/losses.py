"""Losses, one per head. All operate on the head's output dict and target (B,2).

- mse_loss          : baseline regression.
- gaussian_nll_loss : heteroscedastic Gaussian negative log-likelihood.
- evidential_loss   : Deep Evidential Regression NLL + evidence regularizer.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def mse_loss(out: dict, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(out["mean"], target)


def gaussian_nll_loss(out: dict, target: torch.Tensor) -> torch.Tensor:
    mean, var = out["mean"], out["var"]
    return 0.5 * (torch.log(var) + (target - mean) ** 2 / var).mean()


def evidential_loss(out: dict, target: torch.Tensor, lam: float = 0.01) -> torch.Tensor:
    """NIG negative log-likelihood + evidence regularizer (Amini et al., 2020).

    The regularizer penalizes evidence on inaccurate predictions, which is what
    drives epistemic uncertainty up off-distribution instead of staying falsely
    confident.
    """
    gamma, nu = out["gamma"], out["nu"]
    alpha, beta = out["alpha"], out["beta"]
    y = target

    # Negative log-likelihood of the Student-t marginal (NIG evidence).
    omega = 2.0 * beta * (1.0 + nu)
    nll = (
        0.5 * torch.log(math.pi / nu)
        - alpha * torch.log(omega)
        + (alpha + 0.5) * torch.log((y - gamma) ** 2 * nu + omega)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )

    # Evidence regularizer: scale error by total evidence (2*nu + alpha).
    reg = torch.abs(y - gamma) * (2.0 * nu + alpha)

    return (nll + lam * reg).mean()


def build_loss(head_kind: str, cfg: dict):
    if head_kind == "regression":
        return lambda out, y: mse_loss(out, y)
    if head_kind == "gaussian":
        return lambda out, y: gaussian_nll_loss(out, y)
    if head_kind == "evidential":
        lam = cfg["train"]["evidential_lambda"]
        return lambda out, y: evidential_loss(out, y, lam)
    raise ValueError(head_kind)

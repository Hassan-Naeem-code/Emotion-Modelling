"""Prediction heads for 2-D valence/arousal regression. Swappable via config.

All heads return a dict with at least `mean` (B,2). Uncertainty-bearing heads
add the parameters needed to recover predictive variance:

- RegressionHead: deterministic point estimate. No uncertainty.
- GaussianHead:   heteroscedastic mean + variance (aleatoric only).
- EvidentialHead: Deep Evidential Regression. Per-dim Normal-Inverse-Gamma
                  params (gamma, nu, alpha, beta) giving BOTH aleatoric and
                  epistemic uncertainty in one forward pass.

VA is bounded to [-1,1] via tanh on the mean. Variances are strictly positive.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

OUT_DIM = 2  # valence, arousal


def _mlp(in_dim: int, out_dim: int, hidden: int = 256) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, out_dim)
    )


class RegressionHead(nn.Module):
    kind = "regression"

    def __init__(self, in_dim: int):
        super().__init__()
        self.net = _mlp(in_dim, OUT_DIM)

    def forward(self, feats):
        mean = torch.tanh(self.net(feats))
        return {"mean": mean}


class GaussianHead(nn.Module):
    """Predicts mean and log-variance per dim. Aleatoric uncertainty."""
    kind = "gaussian"

    def __init__(self, in_dim: int):
        super().__init__()
        self.net = _mlp(in_dim, OUT_DIM * 2)

    def forward(self, feats):
        out = self.net(feats)
        mean = torch.tanh(out[:, :OUT_DIM])
        log_var = out[:, OUT_DIM:]
        var = F.softplus(log_var) + 1e-6
        return {"mean": mean, "var": var}


class EvidentialHead(nn.Module):
    """Deep Evidential Regression (Amini et al., 2020), applied per dimension.

    Outputs the 4 Normal-Inverse-Gamma params per VA dim:
        gamma : predictive mean (here squashed to [-1,1])
        nu    : > 0, pseudo-count of mean observations (epistemic)
        alpha : > 1, shape (epistemic)
        beta  : > 0, scale (aleatoric)

    Predictive moments:
        aleatoric var = beta / (alpha - 1)
        epistemic var = beta / (nu * (alpha - 1))
    """
    kind = "evidential"

    def __init__(self, in_dim: int):
        super().__init__()
        # 4 params x 2 dims.
        self.net = _mlp(in_dim, OUT_DIM * 4)

    def forward(self, feats):
        out = self.net(feats).view(-1, OUT_DIM, 4)
        gamma = torch.tanh(out[..., 0])
        nu = F.softplus(out[..., 1]) + 1e-6
        alpha = F.softplus(out[..., 2]) + 1.0 + 1e-6   # alpha > 1
        beta = F.softplus(out[..., 3]) + 1e-6
        aleatoric = beta / (alpha - 1.0)
        epistemic = beta / (nu * (alpha - 1.0))
        return {
            "mean": gamma,
            "gamma": gamma, "nu": nu, "alpha": alpha, "beta": beta,
            "aleatoric": aleatoric, "epistemic": epistemic,
            "var": aleatoric + epistemic,
        }


def build_head(kind: str, in_dim: int) -> nn.Module:
    return {
        "regression": RegressionHead,
        "gaussian": GaussianHead,
        "evidential": EvidentialHead,
    }[kind](in_dim)

"""Unit tests for the invariants that make this pipeline trustworthy:
calibration, conformal coverage, uncertainty ranking, rPPG quality-gating, and
the config override mechanism. Run with:  python -m pytest tests/ -q

These use numpy/torch only (no real data, no network) so they run in CI.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import apply_overrides, load_config, set_seed
from eval.metrics import (accuracy_metrics, ccc, empirical_coverage,
                          ood_auroc, regression_ece, uncertainty_score)
from eval.selective import aurc, risk_coverage_curve
from models.wrappers import conformal_calibrate, conformal_intervals
from models import rppg


def test_ccc_perfect_and_offset():
    x = np.linspace(-1, 1, 100)
    assert ccc(x, x) > 0.999                 # identical -> CCC ~ 1
    assert ccc(x, x + 0.5) < ccc(x, x)       # a bias penalizes CCC


def test_regression_ece_calibrated_is_low():
    set_seed(0)
    rng = np.random.default_rng(0)
    n = 5000
    target = rng.normal(0, 1, (n, 2)).astype(np.float32)
    mean = np.zeros((n, 2), dtype=np.float32)
    var = np.ones((n, 2), dtype=np.float32)   # truly N(0,1) residuals
    ece_good = regression_ece(mean, var, target)
    # Overconfident model (variance too small) should be clearly worse.
    ece_bad = regression_ece(mean, var * 0.1, target)
    assert ece_good < 0.05
    assert ece_bad > ece_good


def test_conformal_coverage_holds_on_calibration_distribution():
    rng = np.random.default_rng(1)
    n = 4000
    mean = np.zeros((n, 2), dtype=np.float32)
    var = np.ones((n, 2), dtype=np.float32)
    target = rng.normal(0, 1, (n, 2)).astype(np.float32)
    cal = {"mean": mean[:2000], "var": var[:2000], "target": target[:2000]}
    test = {"mean": mean[2000:], "var": var[2000:], "target": target[2000:]}
    calib = conformal_calibrate(cal, alpha=0.1)
    iv = conformal_intervals(test, calib)
    cov = empirical_coverage(iv["lower"], iv["upper"], test["target"])
    assert 0.86 <= cov <= 0.94               # ~90% target with sampling slack


def test_uncertainty_ranks_errors_so_rejection_lowers_risk():
    # Larger predicted variance should accompany larger errors -> risk drops as
    # we reject the most uncertain. Construct that relationship explicitly.
    rng = np.random.default_rng(2)
    n = 1000
    err = rng.random(n)
    mean = np.zeros((n, 2), dtype=np.float32)
    target = np.stack([err, err], axis=1).astype(np.float32)
    var = np.stack([err, err], axis=1).astype(np.float32) + 1e-3
    cov, risk = risk_coverage_curve(mean, var, target)
    assert risk[0] < risk[-1]                # confident subset has lower risk
    assert aurc(mean, var, target) < risk[-1]


def test_ood_auroc_separates_high_uncertainty():
    var_id = np.full((500, 2), 0.1, dtype=np.float32)
    var_ood = np.full((500, 2), 1.0, dtype=np.float32)
    assert ood_auroc(var_id, var_ood) > 0.99


def test_rppg_recovers_known_hr_and_gates_on_quality():
    t = np.linspace(0, 5, 150)
    clean = np.stack([0.5 * np.ones_like(t),
                      0.5 + 0.02 * np.sin(2 * np.pi * 1.2 * t),
                      0.5 * np.ones_like(t)], axis=1)
    est = rppg.estimate(clean, fps=30)
    assert abs(est["hr_bpm"] - 72) < 5       # 1.2 Hz -> 72 bpm
    assert est["quality"] > 0.5
    # Pure noise -> low quality -> arousal feature gated toward 0.
    noise = np.random.default_rng(3).random((150, 3)).astype(np.float32)
    noisy = rppg.estimate(noise, fps=30)
    assert noisy["quality"] < est["quality"]
    assert abs(noisy["arousal_feature"]) <= abs(est["arousal_feature"]) + 0.2


def test_config_overrides_and_inheritance():
    cfg = load_config("configs/evidential.yaml")
    assert cfg["model"]["head"] == "evidential"      # from evidential.yaml
    assert cfg["data"]["image_size"] == 224          # inherited from _base
    cfg = apply_overrides(cfg, ["data.name=affectnet", "train.epochs=7",
                                "model.pretrained=false"])
    assert cfg["data"]["name"] == "affectnet"
    assert cfg["train"]["epochs"] == 7 and isinstance(cfg["train"]["epochs"], int)
    assert cfg["model"]["pretrained"] is False


def test_evidential_head_outputs_valid_uncertainty():
    from models.heads import EvidentialHead
    set_seed(0)
    head = EvidentialHead(16)
    out = head(torch.randn(8, 16))
    assert out["mean"].shape == (8, 2)
    assert torch.all(out["alpha"] > 1)               # NIG validity
    assert torch.all(out["nu"] > 0) and torch.all(out["beta"] > 0)
    assert torch.all(out["var"] > 0)
    assert torch.all(out["mean"].abs() <= 1.0)       # VA bounded to [-1,1]

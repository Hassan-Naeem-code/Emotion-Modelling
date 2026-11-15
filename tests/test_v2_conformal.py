"""Checks of the paper's central claims on synthetic data where the truth is known.

1. The max-score region (Proposition 1) covers the 2-D point with probability
   >= 1 - alpha under exchangeability, averaged over many random splits.
2. Per-axis intervals at 1 - alpha each cover the point with probability below
   1 - alpha (the diagnosis), and Bonferroni is valid but larger than max-score.
3. A shifted group absent from calibration is under-covered (no guarantee).
"""
import numpy as np

from eval.analyze import ALPHA, conformal


def _data(rng, n, rho=0.5, shift=0.0):
    sd = rng.uniform(0.05, 0.4, size=(n, 2))           # heteroscedastic, known scale
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
    mean = rng.uniform(-0.5, 0.5, size=(n, 2))
    return {"mean": mean, "var": sd ** 2, "target": mean + z * sd + shift}


def test_joint_guarantee_and_per_axis_shortfall():
    rng = np.random.default_rng(0)
    cov = {"joint": [], "perdim_joint": [], "bonf": []}
    area = {"joint": [], "bonf": []}
    for _ in range(200):
        r = conformal(_data(rng, 1000), {"t": _data(rng, 2000)})["t"]
        for k in cov:
            cov[k].append(r[k].mean())
        area["joint"].append(r["joint_area"].mean())
        area["bonf"].append(r["bonf_area"].mean())
    nominal = 1 - ALPHA
    assert np.mean(cov["joint"]) >= nominal - 0.003          # guarantee holds
    assert np.mean(cov["bonf"]) >= nominal - 0.003
    assert np.mean(cov["perdim_joint"]) < nominal - 0.02      # per-axis falls short
    assert np.mean(area["joint"]) < np.mean(area["bonf"])     # and max-score is tighter


def test_unseen_shifted_group_loses_coverage():
    rng = np.random.default_rng(1)
    r = conformal(_data(rng, 1000), {"t": _data(rng, 2000, shift=0.4)})["t"]
    assert r["joint"].mean() < 0.6

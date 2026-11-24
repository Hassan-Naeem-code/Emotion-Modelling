"""Copula-family regions and few-shot recalibration, on synthetic predictions."""
import numpy as np

from eval import joint_regions, recalibrate


def _split(rng, n, rho=0.5, offset=(0.0, 0.0), emotion=None):
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
    sd = rng.uniform(0.05, 0.2, size=(n, 2))
    mean = rng.uniform(-0.5, 0.5, size=(n, 2))
    return {"mean": mean, "var": sd ** 2, "target": mean + z * sd + np.array(offset),
            "emotion": np.zeros(n, int) if emotion is None else emotion}


def test_every_joint_region_covers_and_perdim_does_not():
    rng = np.random.default_rng(0)
    covs = {k: [] for k, _ in joint_regions.REGIONS}
    for _ in range(20):
        m = joint_regions.evaluate(_split(rng, 2000), _split(rng, 4000))
        for k in covs:
            covs[k].append(m[f"cov_{k}"])
    mean = {k: np.mean(v) for k, v in covs.items()}
    assert mean["perdim"] < 0.86
    for k in ("bonf", "gauss", "empcop", "max"):
        assert mean[k] > 0.89, k
    assert 0.05 < m["rho"] < 0.5   # dependence of |residuals| is weaker than of residuals


def test_recalibration_restores_coverage_on_an_offset_group():
    rng = np.random.default_rng(1)
    val = _split(rng, 2000)
    e = np.r_[np.zeros(3000, int), np.ones(600, int)]
    test = _split(rng, 3600, emotion=e)
    test["target"][e == 1] += 0.4           # the held-out group is shifted
    r = recalibrate.run(val, test, held=1, rng=rng)
    assert r["global_cov"] < 0.3
    for k in (20, 50, 100):
        assert r[f"mondrian_k{k}_cov"] > 0.88
        assert r[f"shift_k{k}_cov"] > 0.88
    assert r["shift_k100_area"] < r["mondrian_k100_area"]

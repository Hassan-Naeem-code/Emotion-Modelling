"""Joint VA regions compared with the copula family (no GPU).

    python -m eval.joint_regions --preds results/v2/preds --prefix v2 \
        --out results/summary/joint_regions [--tables paper_v2/tables --name copula]

Every region here is a rectangle built from the same normalised residuals
s_d = |y_d - mu_d| / sigma_d on the calibration split; they differ only in how
the two per-axis thresholds (q_v, q_a) are chosen:

  per-axis      each axis at 1-alpha (no joint guarantee; the common practice)
  Bonferroni    each axis at 1-alpha/2 (valid, conservative)
  independence  each axis at sqrt(1-alpha): copula assuming independent axes
                (Messoudi et al., 2021), valid only if the axes are independent
  Gaussian cop. each axis at level t with C_rho(t, t) = 1-alpha, rho estimated
                from the normal scores of the calibration ranks
  empirical cop. each axis at level t, t the conformal quantile of the max of
                the two calibration ranks (Messoudi et al.'s empirical copula)
  max-score     one threshold on max_d s_d (ours; finite-sample joint guarantee)

The empirical-copula region uses ranks computed on the calibration set itself,
so its guarantee is approximate (error O(1/n)); max-score is exact.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from eval.analyze import ALPHA, EMOTIONS, UNSEEN, load_split, scale_of, _q

REGIONS = [("perdim", "Per-axis"), ("bonf", "Bonferroni"), ("indep", "Indep.\\ copula"),
           ("gauss", "Gaussian copula"), ("empcop", "Empirical copula"), ("max", "Max-score (ours)")]


def _level_q(s: np.ndarray, level: float) -> float:
    """Conformal-style quantile of s at `level` (finite-sample corrected)."""
    n = len(s)
    return float(np.quantile(s, min(1.0, math.ceil((n + 1) * level) / n)))


def _gauss_level(rho: float, alpha: float = ALPHA) -> float:
    """t with P(U1 <= t, U2 <= t) = 1-alpha under a Gaussian copula."""
    mvn = stats.multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]])
    lo, hi = 1 - alpha, 1 - alpha / 2 + 1e-9
    for _ in range(60):
        mid = (lo + hi) / 2
        z = stats.norm.ppf(mid)
        lo, hi = (mid, hi) if mvn.cdf([z, z]) < 1 - alpha else (lo, mid)
    return hi


def thresholds(s_cal: np.ndarray) -> dict[str, np.ndarray]:
    n = len(s_cal)
    out = {
        "perdim": np.array([_level_q(s_cal[:, d], 1 - ALPHA) for d in range(2)]),
        "bonf": np.array([_level_q(s_cal[:, d], 1 - ALPHA / 2) for d in range(2)]),
        "indep": np.array([_level_q(s_cal[:, d], math.sqrt(1 - ALPHA)) for d in range(2)]),
    }
    ranks = np.column_stack([stats.rankdata(s_cal[:, d]) for d in range(2)]) / (n + 1)
    rho = float(np.corrcoef(stats.norm.ppf(ranks).T)[0, 1])
    t = _gauss_level(rho)
    out["gauss"] = np.array([_level_q(s_cal[:, d], t) for d in range(2)])
    t_emp = _q(ranks.max(1) * (n + 1) / n)
    srt = np.sort(s_cal, 0)
    idx = min(n - 1, max(0, math.ceil(t_emp * n) - 1))
    out["empcop"] = srt[idx]
    q = _q(s_cal.max(1))
    out["max"] = np.array([q, q])
    out["_rho"] = np.array([rho])
    return out


def evaluate(val: dict, test: dict) -> dict:
    s_cal = np.abs(val["target"] - val["mean"]) / scale_of(val)
    sc = scale_of(test)
    s = np.abs(test["target"] - test["mean"]) / sc
    e = test["emotion"]
    seen = e != UNSEEN
    th = thresholds(s_cal)
    m = {"rho": float(th["_rho"][0])}
    for key, _ in REGIONS:
        q = th[key]
        cov = (s <= q).all(1)
        m[f"cov_{key}"] = float(cov[seen].mean())
        m[f"area_{key}"] = float(np.prod(2 * q * sc, 1)[seen].mean())
        if "group" in test:
            g = test["group"]
            m[f"clip_{key}"] = float(np.mean([cov[g == k].mean() for k in np.unique(g)]))
        groups = [k for k in range(UNSEEN) if (e == k).any()]
        if groups:
            m[f"worst_{key}"] = float(min(cov[e == k].mean() for k in groups))
    return m


def ensemble_runs(seeds: dict) -> dict:
    from eval.analyze import ensemble
    return {sp: ensemble([seeds[s][sp] for s in sorted(seeds)]) for sp in ("val", "test")}


def latex(table: dict, rows: list[tuple[str, str]]) -> str:
    """Rows = methods; columns = regions (joint coverage / area relative to max-score)."""
    has_worst = all(f"worst_max" in table[k] for k, _ in rows)
    out = [r"\begin{tabular}{l" + "r" * len(REGIONS) + "}", r"\toprule",
           "Joint coverage (\\%) & " + " & ".join(n for _, n in REGIONS) + r" \\", r"\midrule"]
    for k, name in rows:
        r = table[k]
        out.append(name + " & " + " & ".join(f"{100 * r[f'cov_{x}'][0]:.1f}" for x, _ in REGIONS) + r" \\")
    out += [r"\midrule", r"\multicolumn{" + str(len(REGIONS) + 1) + r"}{l}{\emph{Area relative to max-score}} \\"]
    for k, name in rows:
        r = table[k]
        out.append(name + " & " + " & ".join(
            f"{r[f'area_{x}'][0] / r['area_max'][0]:.2f}" for x, _ in REGIONS) + r" \\")
    if has_worst:
        out += [r"\midrule", r"\multicolumn{" + str(len(REGIONS) + 1) + r"}{l}{\emph{Worst seen expression, joint coverage (\%)}} \\"]
        for k, name in rows:
            r = table[k]
            out.append(name + " & " + " & ".join(f"{100 * r[f'worst_{x}'][0]:.1f}" for x, _ in REGIONS) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--prefix", default="v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tables", default=None)
    ap.add_argument("--name", default="copula", help="table file name (without .tex)")
    args = ap.parse_args()

    runs: dict = defaultdict(dict)
    for f in sorted(Path(args.preds).glob(f"{args.prefix}_*.npz")):
        mt = re.match(rf"{re.escape(args.prefix)}_(.+)_s(\d+)$", f.stem)
        if not mt:
            continue
        npz = np.load(f)
        runs[mt[1]][mt[2]] = {s: load_split(npz, s) for s in ("val", "test")}

    results = {}
    for method, seeds in sorted(runs.items()):
        results[method] = [evaluate(sp["val"], sp["test"]) for sp in seeds.values()]
        if method in ("gaussian_nll", "evidential") and len(seeds) >= 2:
            ens = ensemble_runs(seeds)
            results[f"{method}_ensemble"] = [evaluate(ens["val"], ens["test"])]

    table = {m: {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs])))
                 for k in rs[0]} for m, rs in results.items()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps(table, indent=2))

    print(f"{'method':22s} rho   " + "  ".join(f"{k:>14s}" for k, _ in REGIONS))
    for m, r in table.items():
        print(f"{m:22s} {r['rho'][0]:+.2f} " + "  ".join(
            f"{100*r[f'cov_{k}'][0]:5.1f}/{r[f'area_{k}'][0]/r['area_max'][0]:4.2f}"
            + (f"/{100*r[f'worst_{k}'][0]:4.1f}" if f"worst_{k}" in r else "")
            for k, _ in REGIONS))

    rows = [(k, n) for k, n in [("baseline", "Deterministic"), ("gaussian_nll", "Gaussian"),
                                ("evidential", "Evidential"), ("mcdropout", "MC-dropout"),
                                ("gaussian_nll_ensemble", "Gaussian ens."),
                                ("evidential_ensemble", "Evidential ens.")] if k in table]
    if args.tables:
        Path(args.tables).mkdir(parents=True, exist_ok=True)
        (Path(args.tables) / f"{args.name}.tex").write_text(latex(table, rows))


if __name__ == "__main__":
    main()

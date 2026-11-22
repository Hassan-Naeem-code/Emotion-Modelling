"""v2 analysis: every resubmission table, from saved predictions (no GPU).

    python -m eval.analyze --preds results/preds/v2 --out results/summary/v2

Reads <preds>/<method>_s<seed>.npz written by eval/dump_preds.py and reports,
per method, mean +/- std over seeds. Deep ensembles are formed here from the
saved seeds of a method (a K-member ensemble needs no extra training).

What it measures, and why each matters for the paper:
  accuracy      CCC / RMSE per dimension.
  proper scores NLL and CRPS (Gaussian moment-matched; for the evidential head
                the Student-t marginal is used for NLL). ECE alone can be gamed.
  coverage      split conformal at 1-alpha, calibrated on Validation:
                  * per-dim  : the v1 method; we report its JOINT coverage, i.e.
                               how often BOTH intervals hold, which is what a VA
                               point estimate actually needs.
                  * joint    : one score max_d |r_d|/sigma_d, giving a rectangle
                               with a finite-sample joint guarantee.
                and per-emotion coverage on the seen classes (worst group).
  novelty       AffectNet 'contempt' is absent from Train and Validation, so it
                is a natural unseen-expression set: coverage on it, and AUROC of
                the uncertainty for flagging it.
  shift         corruption shift: joint coverage and OOD AUROC.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import special, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import accuracy_metrics, regression_ece  # noqa: E402
from eval.selective import aurc  # noqa: E402

EMOTIONS = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "anger", "contempt"]
UNSEEN = 7          # contempt: not in Train or Validation of this AffectNet release
ALPHA = 0.1
EPS = 1e-6


# ----------------------------------------------------------------- scoring ---

def _q(scores: np.ndarray, alpha: float = ALPHA) -> float:
    n = len(scores)
    return float(np.quantile(scores, min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)))


def auroc(neg: np.ndarray, pos: np.ndarray) -> float:
    """P(score_pos > score_neg), ties counted half (Mann-Whitney)."""
    s = np.concatenate([neg, pos])
    r = stats.rankdata(s)
    n_p, n_n = len(pos), len(neg)
    return float((r[n_n:].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


def gaussian_nll(mean, var, y) -> float:
    var = np.maximum(var, EPS)
    return float(np.mean(0.5 * np.log(2 * np.pi * var) + (y - mean) ** 2 / (2 * var)))


def student_t_nll(p: dict, y) -> float:
    """NLL of the evidential head's own predictive (Student-t, 2*alpha dof)."""
    g, nu, a, b = p["mean"], p["nu"], p["alpha"], p["beta"]
    scale2 = b * (1 + nu) / (nu * a)
    return float(np.mean(-stats.t.logpdf(y, df=2 * a, loc=g, scale=np.sqrt(scale2))))


def gaussian_crps(mean, var, y) -> float:
    sd = np.sqrt(np.maximum(var, EPS))
    z = (y - mean) / sd
    crps = sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    return float(np.mean(crps))


# ------------------------------------------------------------- conformal ---

def scale_of(p: dict) -> np.ndarray:
    """Per-sample, per-dim scale for normalized scores. A model with no
    uncertainty gets a constant scale: plain absolute-residual conformal."""
    if np.any(p["var"] > 0):
        return np.sqrt(p["var"]) + EPS
    return np.ones_like(p["mean"])


def conformal(cal: dict, tests: dict[str, dict]) -> dict:
    """Return per-split boolean joint-cover arrays and areas for both variants."""
    s_cal = np.abs(cal["target"] - cal["mean"]) / scale_of(cal)
    q_dim = np.array([_q(s_cal[:, d]) for d in range(2)])
    q_joint = _q(s_cal.max(1))
    # Bonferroni: each dim at alpha/2. Also a valid joint guarantee, but
    # conservative; the joint max-score should match its coverage with less area.
    q_bonf = np.array([_q(s_cal[:, d], ALPHA / 2) for d in range(2)])
    out = {}
    for name, p in tests.items():
        sc = scale_of(p)
        s = np.abs(p["target"] - p["mean"]) / sc
        out[name] = {
            "perdim_marg": (s <= q_dim).mean(1),             # per-dim marginal cover
            "perdim_joint": (s <= q_dim).all(1),             # both hold
            "joint": s.max(1) <= q_joint,
            "bonf": (s <= q_bonf).all(1),
            "perdim_area": np.prod(2 * q_dim * sc, 1),
            "joint_area": np.prod(2 * q_joint * sc, 1),
            "bonf_area": np.prod(2 * q_bonf * sc, 1),
        }
    return out


# ------------------------------------------------------------------ io -----

def load_split(npz, split: str) -> dict | None:
    if f"{split}_mean" not in npz:
        return None
    keys = {k[len(split) + 1:] for k in npz.files if k.startswith(split + "_")}
    return {k: npz[f"{split}_{k}"] for k in keys}


def ensemble(members: list[dict]) -> dict:
    """Uniform mixture, moment-matched: mean of means; var = mean var + var of means."""
    means = np.stack([m["mean"] for m in members])
    vars_ = np.stack([m["var"] for m in members])
    out = {"mean": means.mean(0), "var": vars_.mean(0) + means.var(0),
           "epistemic": means.var(0), "target": members[0]["target"],
           "emotion": members[0]["emotion"]}
    if "group" in members[0]:
        out["group"] = members[0]["group"]
    return out


# ------------------------------------------------------------- metrics -----

def evaluate(splits: dict[str, dict], head: str) -> dict:
    val, test = splits["val"], splits["test"]
    has_unc = bool(np.any(test["var"] > 0))
    y, e = test["target"], test["emotion"]
    seen = e != UNSEEN
    m: dict = {}

    acc = accuracy_metrics(test["mean"][seen], y[seen])
    m.update({"ccc_v": acc["ccc_valence"], "ccc_a": acc["ccc_arousal"],
              "ccc": acc["ccc_mean"], "rmse": acc["rmse_mean"]})
    if has_unc:
        ts = {k: v[seen] for k, v in test.items() if hasattr(v, "shape") and len(v) == len(e)}
        m["nll"] = (student_t_nll(ts, ts["target"]) if head == "evidential" and "nu" in ts
                    else gaussian_nll(ts["mean"], ts["var"], ts["target"]))
        m["crps"] = gaussian_crps(ts["mean"], ts["var"], ts["target"])
        m["ece"] = regression_ece(ts["mean"], ts["var"], ts["target"])
        m["aurc"] = aurc(ts["mean"], ts["var"], ts["target"])

    tests = {"test": test}
    if "corrupt" in splits:
        tests["corrupt"] = splits["corrupt"]
    cf = conformal(val, tests)
    t = cf["test"]
    m["cov_perdim_marg"] = float(t["perdim_marg"][seen].mean())
    m["cov_perdim_joint"] = float(t["perdim_joint"][seen].mean())
    m["cov_joint"] = float(t["joint"][seen].mean())
    m["area_perdim"] = float(t["perdim_area"][seen].mean())
    m["area_joint"] = float(t["joint_area"][seen].mean())
    m["cov_bonf"] = float(t["bonf"][seen].mean())
    m["area_bonf"] = float(t["bonf_area"][seen].mean())
    if "group" in test:
        # Video data: frames of one clip are correlated, so also report coverage
        # averaged per clip (each clip counts once), alongside the frame average.
        g = test["group"]
        clip_mean = lambda c: float(np.mean([c[g == k].mean() for k in np.unique(g)]))
        m["cov_joint_clip"] = clip_mean(t["joint"])
        m["cov_perdim_joint_clip"] = clip_mean(t["perdim_joint"])
        m["cov_bonf_clip"] = clip_mean(t["bonf"])
        m["n_clips"] = float(len(np.unique(g)))
    per = {EMOTIONS[k]: float(t["joint"][e == k].mean()) for k in range(8) if (e == k).any()}
    per_v1 = {EMOTIONS[k]: float(t["perdim_joint"][e == k].mean()) for k in range(8) if (e == k).any()}
    if per:                       # datasets without expression labels skip these
        m["cov_joint_by_emotion"] = per
        m["cov_perdim_by_emotion"] = per_v1
        m["worst_seen_joint"] = min(v for k, v in per.items() if k != "contempt")
        m["worst_seen_perdim"] = min(v for k, v in per_v1.items() if k != "contempt")
    if (e == UNSEEN).any():
        m["contempt_cov_joint"] = per["contempt"]
        m["contempt_rmse"] = float(np.sqrt(np.mean((test["mean"][e == UNSEEN] - y[e == UNSEEN]) ** 2)))
        if has_unc:
            u = np.sqrt(test["var"]).sum(1)
            m["contempt_auroc"] = auroc(u[seen], u[e == UNSEEN])
    if "corrupt" in splits:
        c = splits["corrupt"]
        cs = c["emotion"] != UNSEEN
        m["shift_ccc"] = accuracy_metrics(c["mean"][cs], c["target"][cs])["ccc_mean"]
        m["shift_cov_perdim_joint"] = float(cf["corrupt"]["perdim_joint"][cs].mean())
        m["shift_cov_joint"] = float(cf["corrupt"]["joint"][cs].mean())
        if has_unc:
            m["shift_auroc"] = auroc(np.sqrt(test["var"]).sum(1), np.sqrt(c["var"]).sum(1))
            m["shift_ece"] = regression_ece(c["mean"][cs], c["var"][cs], c["target"][cs])
    return m


# --------------------------------------------------------------- paper -----

ROWS = [  # (method key, display name)
    ("v2_baseline", "Deterministic"),
    ("v2_gaussian_nll", "Gaussian"),
    ("v2_evidential", "Evidential"),
    ("v2_mcdropout", "MC-dropout"),
    ("v2_gaussian_nll_ensemble5", "Gaussian ens."),
    ("v2_evidential_ensemble5", "Evidential ens."),
]


def _cell(row, key, best=None, digits=3):
    if key not in row:
        return "--"
    mu, sd = row[key]
    txt = f"{mu:.{digits}f}"
    if best is not None and abs(mu - best) < 10 ** -digits / 2:
        txt = rf"\textbf{{{txt}}}"
    if row["n"] > 1:
        txt += rf"$_{{\pm{sd:.{digits}f}}}$".replace("0.", ".", 1)
    return txt


def _tabular(table, spec, header, cols):
    """cols: list of (metric key, lower_is_better or None)."""
    rows = [(k, n) for k, n in ROWS if k in table]
    best = {}
    for key, low in cols:
        vals = [table[k][key][0] for k, _ in rows if key in table[k]]
        if low is not None and vals:
            best[key] = min(vals) if low else max(vals)
    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", header + r" \\", r"\midrule"]
    for k, name in rows:
        lines.append(" & ".join([name] + [_cell(table[k], key, best.get(key))
                                          for key, _ in cols]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _wilson(p, n, z=1.96):
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def write_paper_tables(table, results, d: Path):
    d.mkdir(parents=True, exist_ok=True)
    (d / "main.tex").write_text(_tabular(
        table, "lccccccc",
        r"Method & CCC-V$\uparrow$ & CCC-A$\uparrow$ & RMSE$\downarrow$ & "
        r"NLL$\downarrow$ & CRPS$\downarrow$ & ECE$\downarrow$ & AURC$\downarrow$",
        [("ccc_v", False), ("ccc_a", False), ("rmse", True), ("nll", True),
         ("crps", True), ("ece", True), ("aurc", True)]))
    (d / "coverage.tex").write_text(_tabular(
        table, "lcccccccc",
        r" & \multicolumn{2}{c}{Per-axis} & \multicolumn{2}{c}{Bonferroni} & "
        r"\multicolumn{2}{c}{Max-score (ours)} & \multicolumn{2}{c}{Worst / Shift (ours)} \\"
        "\n" r"Method & Joint & Area & Joint & Area & Joint & Area & Worst & Shift",
        [("cov_perdim_joint", None), ("area_perdim", None), ("cov_bonf", None),
         ("area_bonf", None), ("cov_joint", None), ("area_joint", True),
         ("worst_seen_joint", False), ("shift_cov_joint", None)]))
    n_con = 70
    t = _tabular(table, "lccc",
                 r"Method & RMSE$\downarrow$ & Joint cov. & AUROC$\uparrow$",
                 [("contempt_rmse", True), ("contempt_cov_joint", None),
                  ("contempt_auroc", False)])
    lo, hi = _wilson(0.4, n_con)
    t += rf"% Wilson 95% half-width at p=0.4, n={n_con}: {(hi-lo)/2:.3f}" + "\n"
    (d / "unseen.tex").write_text(t)

    # Per-expression joint coverage figure (evidential, averaged over seeds).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = results.get("v2_evidential")
    if r is None:
        return
    seeds = list(r["seeds"].values())
    names = EMOTIONS
    per = np.array([[s["cov_perdim_by_emotion"].get(e, np.nan) for e in names] for s in seeds])
    joint = np.array([[s["cov_joint_by_emotion"].get(e, np.nan) for e in names] for s in seeds])
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.bar(x - 0.2, per.mean(0), 0.4, yerr=per.std(0), label="Per-axis", color="#9aa5b1")
    ax.bar(x + 0.2, joint.mean(0), 0.4, yerr=joint.std(0), label="Max-score (ours)", color="#2b6cb0")
    ax.axhline(1 - ALPHA, color="k", ls="--", lw=1, label="Nominal 90%")
    ax.set_xticks(x, [n if n != "contempt" else "contempt\n(unseen)" for n in names], fontsize=8)
    ax.set_ylabel("Joint coverage")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(d / "per_emotion.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- main -----

HEADS = {"baseline": "regression", "gaussian_nll": "gaussian", "evidential": "evidential"}


def base_method(method: str) -> str:
    """'v2_evidential', 'v2c_evidential', 'v2_afew_evidential' -> 'evidential'."""
    for name in (*HEADS, "mcdropout"):
        if method.endswith("_" + name) or method == name:
            return name
    return re.sub(r"^v\d+[a-z]*_", "", method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="dir of <method>_s<seed>.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ensemble-of", default="gaussian_nll,evidential",
                    help="methods whose seeds are also combined into a deep ensemble")
    ap.add_argument("--prefix", default="v2",
                    help="method-name prefix for the paper tables, e.g. v2c or v2_afew")
    ap.add_argument("--tables", default=None,
                    help="also write the paper's LaTeX tables + figure into this dir")
    args = ap.parse_args()

    runs: dict[str, dict[str, dict]] = defaultdict(dict)
    for f in sorted(Path(args.preds).glob("*.npz")):
        mt = re.match(r"(.+)_s(\d+)$", f.stem)
        if not mt:
            continue
        npz = np.load(f)
        splits = {s: load_split(npz, s) for s in ("val", "test", "corrupt")}
        runs[mt.group(1)][mt.group(2)] = {k: v for k, v in splits.items() if v is not None}

    results: dict[str, dict] = {}
    for method, seeds in sorted(runs.items()):
        base = base_method(method)
        head = HEADS.get(base, base)
        per_seed = {s: evaluate(sp, head) for s, sp in sorted(seeds.items())}
        results[method] = {"seeds": per_seed, "n": len(per_seed)}
        if base in args.ensemble_of.split(",") and len(seeds) >= 2:
            ens = {sp: ensemble([seeds[s][sp] for s in sorted(seeds)])
                   for sp in ("val", "test", "corrupt") if all(sp in seeds[s] for s in seeds)}
            results[f"{method}_ensemble{len(seeds)}"] = {
                "seeds": {"ens": evaluate(ens, "gaussian")}, "n": 1}

    # mean / std over seeds
    table = {}
    for method, r in results.items():
        keys = [k for k, v in next(iter(r["seeds"].values())).items() if isinstance(v, float)]
        table[method] = {"n": r["n"], **{
            k: (float(np.mean([s[k] for s in r["seeds"].values()])),
                float(np.std([s[k] for s in r["seeds"].values()])))
            for k in keys}}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps({"summary": table, "runs": results}, indent=2))
    cols = ["ccc", "ccc_v", "ccc_a", "nll", "crps", "ece", "aurc", "cov_perdim_joint",
            "area_perdim", "cov_joint", "area_joint", "cov_bonf", "area_bonf", "worst_seen_perdim", "worst_seen_joint",
            "contempt_cov_joint", "contempt_auroc", "shift_ccc", "shift_cov_joint",
            "shift_auroc", "cov_perdim_joint_clip", "cov_joint_clip", "cov_bonf_clip"]
    with open(out / "analysis.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "n"] + cols)
        for method, row in table.items():
            w.writerow([method, row["n"]] + [
                f"{row[c][0]:.4f}±{row[c][1]:.4f}" if c in row else "" for c in cols])
    # console view
    print(f"{'method':28s} n " + " ".join(f"{c[:11]:>11s}" for c in cols))
    for method, row in table.items():
        print(f"{method:28s} {row['n']} " + " ".join(
            f"{row[c][0]:11.3f}" if c in row else f"{'-':>11s}" for c in cols))
    print(f"\nwrote {out/'analysis.json'} and analysis.csv")
    if args.tables:
        if args.prefix != "v2":   # point the table rows at this run's method names
            global ROWS
            ROWS = [(k.replace("v2_", args.prefix + "_", 1), n) for k, n in ROWS]
        write_paper_tables(table, results, Path(args.tables))


if __name__ == "__main__":
    main()

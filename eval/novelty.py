"""Feature-space novelty detectors for the unseen expression (contempt).

    python -m eval.novelty --feats results/v2/feats --preds results/v2/preds \
        --out results/summary/v2_novelty [--tables paper_v2/tables]

For every model with a feature file, scores each test face with:
  sigma      total predictive std (sum over axes), the baseline from the paper
  maha       Mahalanobis distance to one Gaussian fit to training features
  maha_cls   min class-conditional Mahalanobis, shared covariance (Lee et al.)
  knn        distance to the k-th nearest training feature, L2-normalised
             features (Sun et al.), k=50

Reports AUROC (contempt vs seen test), and the operating point a deployment
would use: threshold at the 95th percentile of *validation* scores (seen data,
never test), giving the fraction of contempt flagged (TPR) and of seen test
flagged (FPR). Then the coverage that results if flagged inputs abstain:
joint coverage of the evidential max-score region on contempt faces that are
*not* flagged, and the share of contempt that is either flagged or covered.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from eval.analyze import auroc

CONTEMPT = 7
SEEN = list(range(7))


def _shrunk_inv(x: np.ndarray, shrink: float = 0.1) -> np.ndarray:
    cov = np.cov(x, rowvar=False)
    d = cov.shape[0]
    cov = (1 - shrink) * cov + shrink * np.trace(cov) / d * np.eye(d)
    return np.linalg.inv(cov)


def _maha(train, q, inv, mu):
    z = q - mu
    return ((z @ inv) * z).sum(1)


def scores(tr, tr_emo, splits: dict, k: int = 50) -> dict:
    tr = tr.astype(np.float64)
    keep = np.isin(tr_emo, SEEN)
    tr, tr_emo = tr[keep], tr_emo[keep]
    mu = tr.mean(0)
    inv = _shrunk_inv(tr)
    cls = [c for c in SEEN if (tr_emo == c).sum() > 10]
    mus = {c: tr[tr_emo == c].mean(0) for c in cls}
    inv_w = _shrunk_inv(np.concatenate([tr[tr_emo == c] - mus[c] for c in cls]))
    trn = tr / np.linalg.norm(tr, axis=1, keepdims=True)
    out = {}
    for name, q in splits.items():
        q = q.astype(np.float64)
        s = {"maha": _maha(tr, q, inv, mu),
             "maha_cls": np.min([_maha(tr, q, inv_w, mus[c]) for c in cls], 0)}
        qn = q / np.linalg.norm(q, axis=1, keepdims=True)
        d = []
        for i in range(0, len(qn), 1000):
            sim = qn[i:i + 1000] @ trn.T
            d.append(1 - np.partition(sim, -k, axis=1)[:, -k])
        s["knn"] = np.concatenate(d)
        out[name] = s
    return out


def joint_region(pred, alpha=0.1):
    """Max-score conformal region from the dumped preds (val = calibration)."""
    def sc(p):
        sd = np.sqrt(np.maximum(pred[f"{p}_var"], 1e-12))
        if not np.any(pred[f"{p}_var"] > 0):
            sd = np.ones_like(sd)
        return (np.abs(pred[f"{p}_target"] - pred[f"{p}_mean"]) / sd).max(1)
    cal = sc("val")
    n = len(cal)
    q = np.sort(cal)[min(n - 1, int(np.ceil((n + 1) * (1 - alpha))) - 1)]
    return {p: sc(p) <= q for p in ("test", "corrupt")}


def evaluate(feat, pred) -> dict:
    s = scores(feat["train_feat"], feat["train_emotion"],
               {p: feat[f"{p}_feat"] for p in ("val", "test", "corrupt")})
    for p in ("val", "test", "corrupt"):
        s[p]["sigma"] = np.sqrt(np.maximum(pred[f"{p}_var"], 0)).sum(1)
    emo = feat["test_emotion"]
    assert np.array_equal(emo, pred["test_emotion"]), "feature/pred order mismatch"
    is_c = emo == CONTEMPT
    seen = np.isin(emo, SEEN)
    covered = joint_region(pred)["test"]
    res = {}
    for det in ("sigma", "maha", "maha_cls", "knn"):
        if not np.any(s["test"][det] > 0):
            continue
        t, v = s["test"][det], s["val"][det]
        thr = np.quantile(v, 0.95)
        flag = t > thr
        res[det] = {
            "auroc": auroc(t[seen], t[is_c]),
            "tpr": float(flag[is_c].mean()),
            "fpr": float(flag[seen].mean()),
            "contempt_cov_kept": float(covered[is_c & ~flag].mean())
            if (is_c & ~flag).any() else float("nan"),
            "contempt_safe": float((flag | covered)[is_c].mean()),
            "seen_safe": float((flag | covered)[seen].mean()),
            "shift_auroc": auroc(t, s["corrupt"][det]),
        }
    res["contempt_cov_no_abstain"] = float(covered[is_c].mean())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tables", default=None)
    a = ap.parse_args()
    runs: dict = {}
    for f in sorted(Path(a.feats).glob("*.npz")):
        m = re.match(r"(.+)_s(\d+)$", f.stem)
        method, seed = m.group(1), m.group(2)
        pf = Path(a.preds) / f.name
        if not pf.exists():
            continue
        r = evaluate(np.load(f), np.load(pf))
        runs.setdefault(method, {})[seed] = r
        print(f.stem, {k: (round(v["auroc"], 3) if isinstance(v, dict) else round(v, 3))
                       for k, v in r.items()}, flush=True)
    summary = {}
    for method, seeds in runs.items():
        rs = list(seeds.values())
        summary[method] = {"n": len(rs)}
        for det in rs[0]:
            if isinstance(rs[0][det], dict):
                summary[method][det] = {k: [float(np.mean([r[det][k] for r in rs])),
                                            float(np.std([r[det][k] for r in rs]))]
                                        for k in rs[0][det]}
            else:
                summary[method][det] = [float(np.mean([r[det] for r in rs])),
                                        float(np.std([r[det] for r in rs]))]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    Path(a.out, "novelty.json").write_text(json.dumps({"summary": summary, "runs": runs},
                                                      indent=1))
    for method, sm in summary.items():
        print(f"\n{method} (n={sm['n']}); no-abstain contempt cov "
              f"{sm['contempt_cov_no_abstain'][0]:.3f}")
        for det, v in sm.items():
            if isinstance(v, dict):
                print(f"  {det:9s} " + " ".join(f"{k}={x[0]:.3f}±{x[1]:.3f}"
                                                 for k, x in v.items()))
    if a.tables:
        write_table(summary, Path(a.tables) / "novelty.tex")


NAMES = {"sigma": "Predictive $\\sum_d\\hat\\sigma_d$", "maha": "Mahalanobis",
         "maha_cls": "Class-cond.\\ Mahalanobis", "knn": "$k$-NN ($k{=}50$)"}


def write_table(summary, path):
    m = summary.get("v2_evidential")
    if m is None:
        return
    rows = []
    for det in ("sigma", "maha", "maha_cls", "knn"):
        if det not in m:
            continue
        v = m[det]
        f = lambda k: f"{v[k][0]:.3f}$_{{\\pm.{round(v[k][1] * 1000):03d}}}$"
        rows.append(f"{NAMES[det]} & {f('auroc')} & {f('tpr')} & {f('fpr')} & "
                    f"{f('contempt_safe')} & {f('shift_auroc')} \\\\")
    path.write_text(
        "\\begin{tabular}{lccccc}\n\\toprule\n"
        "Detector & AUROC$\\uparrow$ & Flagged contempt & Flagged seen & "
        "Contempt safe & Shift AUROC \\\\\n\\midrule\n" + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

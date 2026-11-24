"""How many labelled examples of a new expression restore the guarantee? (no GPU)

    python -m eval.recalibrate --preds results/loeo/preds \
        --out results/summary/recalibrate [--tables paper_v2/tables]

For every leave-one-expression-out run, the held-out expression has no
calibration data, so its coverage collapses (eval/loeo.py). Here we give the
calibrator k labelled examples of that expression, drawn at random from its
test images, and evaluate on the rest of them (repeated draws):

  global      the original max-score threshold (no group data; the failure)
  mondrian    a separate max-score threshold for the new expression, from its
              k examples (group-conditional / Mondrian conformal). For k >= 9
              this has finite-sample coverage >= 1-alpha on that expression.
  shift       half the k examples estimate the expression's mean VA offset,
              which is subtracted from the predictions; the other half
              calibrate the max-score threshold on the corrected residuals.
              Also exact (the halves are disjoint), and much smaller when the
              failure is a systematic offset, as it is here.

Area is reported relative to the region the seen expressions receive.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval.analyze import ALPHA, EMOTIONS, UNSEEN, load_split, scale_of, _q

NAME = re.compile(r"loeo_(?P<method>.+)_h(?P<held>\d+)_s(?P<seed>\d+)\.npz$")
KS = (10, 20, 50, 100)
TABLE_KS = (20, 50, 100)   # k=10 leaves 5 per half: too few for the shift variant
REPS = 200


def _qk(s: np.ndarray) -> float:
    """Conformal quantile; inf when there are too few points for 1-alpha."""
    n = len(s)
    r = math.ceil((n + 1) * (1 - ALPHA))
    return float(np.sort(s)[r - 1]) if r <= n else float("inf")


def run(val: dict, test: dict, held: int, rng: np.random.Generator) -> dict:
    e = test["emotion"]
    H = np.flatnonzero(e == held)
    seen = (e != held) & (e != UNSEEN) & (e >= 0)
    sc = scale_of(test)
    res = test["target"] - test["mean"]
    s = np.abs(res) / sc
    q = _q((np.abs(val["target"] - val["mean"]) / scale_of(val)).max(1))
    seen_area = float(np.prod(2 * q * sc[seen], 1).mean())
    out = {"n": len(H), "global_cov": float((s[H].max(1) <= q).mean())}
    for k in KS:
        if k > len(H) - 20:
            continue
        cov = {"mondrian": [], "shift": []}
        area = {"mondrian": [], "shift": []}
        for _ in range(REPS):
            perm = rng.permutation(H)
            cal, ev = perm[:k], perm[k:]
            qm = _qk(s[cal].max(1))
            cov["mondrian"].append((s[ev].max(1) <= qm).mean())
            area["mondrian"].append(np.prod(2 * qm * sc[ev], 1).mean() / seen_area)
            a, b = cal[: k // 2], cal[k // 2:]
            off = res[a].mean(0)
            qs = _qk((np.abs(res[b] - off) / sc[b]).max(1))
            cov["shift"].append((((np.abs(res[ev] - off) / sc[ev]).max(1)) <= qs).mean())
            area["shift"].append(np.prod(2 * qs * sc[ev], 1).mean() / seen_area)
        for m in cov:
            fin = np.isfinite(area[m])
            out[f"{m}_k{k}_cov"] = float(np.mean(cov[m]))
            out[f"{m}_k{k}_area"] = float(np.median(np.array(area[m])[fin])) if fin.any() else float("inf")
    return out


def latex(rows: dict, method: str) -> str:
    out = [r"\begin{tabular}{lrr" + "rr" * len(TABLE_KS) + "}", r"\toprule",
           r" & & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{$k={k}$}}" for k in TABLE_KS) + r" \\",
           "Held out & $n$ & None & " + " & ".join(["Group", "+shift"] * len(TABLE_KS)) + r" \\", r"\midrule"]

    def cell(r, m, k):
        c, a = r.get(f"{m}_k{k}_cov"), r.get(f"{m}_k{k}_area")
        if c is None or not math.isfinite(a):
            return "--"
        return f"{100 * c:.0f} ({a:.1f}$\\times$)"

    for held in sorted(h for (m, h) in rows if m == method):
        r = rows[(method, held)]
        cells = [EMOTIONS[held], str(int(r["n"])), f"{100 * r['global_cov']:.0f}"]
        for k in TABLE_KS:
            cells += [cell(r, "mondrian", k), cell(r, "shift", k)]
        out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tables", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    per = defaultdict(list)
    for f in sorted(Path(args.preds).glob("loeo_*.npz")):
        mt = NAME.search(f.name)
        if not mt:
            continue
        npz = np.load(f)
        per[(mt["method"], int(mt["held"]))].append(
            run(load_split(npz, "val"), load_split(npz, "test"), int(mt["held"]), rng))

    rows = {key: {k: float(np.mean([r[k] for r in rs if k in r])) for k in rs[0]}
            for key, rs in per.items()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "recalibrate.json").write_text(json.dumps(
        {f"{m}_{EMOTIONS[h]}": r for (m, h), r in rows.items()}, indent=2))
    for (m, h), r in sorted(rows.items()):
        print(f"{m:11s} {EMOTIONS[h]:9s} n={r['n']:5.0f} none={100*r['global_cov']:5.1f}  " + "  ".join(
            f"k={k}: grp {100*r[f'mondrian_k{k}_cov']:4.1f}/{r[f'mondrian_k{k}_area']:4.2f}x "
            f"shift {100*r[f'shift_k{k}_cov']:4.1f}/{r[f'shift_k{k}_area']:4.2f}x"
            for k in KS if f"mondrian_k{k}_cov" in r))
    if args.tables:
        Path(args.tables).mkdir(parents=True, exist_ok=True)
        (Path(args.tables) / "recalibrate.tex").write_text(latex(rows, "evidential"))


if __name__ == "__main__":
    main()

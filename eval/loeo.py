"""Leave-one-expression-out (LOEO): does the contempt result generalise?

The main paper's unseen-expression evidence is AffectNet 'contempt', which the
release we use leaves out of Train and Validation. That is one expression and a
small test set. Here each of the seven *seen* expressions is held out in turn:
removed from Train and from Validation (the conformal calibration set) via
`data.holdout_expression=k`, and kept in Test. Contempt stays unseen in every
run, so it is excluded from the "seen" group as well.

    python -m eval.loeo --preds results/loeo/preds --out results/summary/loeo \
        [--ref results/v2/preds]

Reads <preds>/loeo_<method>_h<k>_s<seed>.npz (written by scripts/run_loeo.sh).
With --ref, also reads the matching v2_<method>_s<seed>.npz from the main runs
and reports each expression's joint coverage when it WAS in training, so the
table shows the drop caused by holding it out, not just the level.

Per (method, held-out expression), mean +/- std over seeds:
  n_held          test images of the held-out expression
  seen_cov_*      coverage on the remaining seen expressions (should stay ~90%)
  held_cov_*      coverage on the held-out expression (per-axis, max-score, Bonferroni)
  held_cov_ref    max-score coverage on that expression in the main runs (--ref)
  held_auroc      AUROC of predicted uncertainty for flagging held-out vs seen
  held_rmse       RMSE on the held-out expression
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.analyze import EMOTIONS, UNSEEN, auroc, conformal, load_split  # noqa: E402

NAME = re.compile(r"loeo_(?P<method>.+)_h(?P<held>\d+)_s(?P<seed>\d+)\.npz$")
METHODS = {"evidential": "Evidential", "gaussian_nll": "Gaussian",
           "mcdropout": "MC-dropout", "baseline": "Deterministic"}


def evaluate(val: dict, test: dict, held: int) -> dict:
    e = test["emotion"]
    is_held = e == held
    seen = (e != held) & (e != UNSEEN) & (e >= 0)
    t = conformal(val, {"test": test})["test"]
    m = {
        "n_held": int(is_held.sum()),
        "seen_cov_perdim": float(t["perdim_joint"][seen].mean()),
        "seen_cov_joint": float(t["joint"][seen].mean()),
        "held_cov_perdim": float(t["perdim_joint"][is_held].mean()),
        "held_cov_joint": float(t["joint"][is_held].mean()),
        "held_cov_bonf": float(t["bonf"][is_held].mean()),
        "held_rmse": float(np.sqrt(np.mean((test["mean"][is_held] - test["target"][is_held]) ** 2))),
        # Where the held-out faces are, and where the model puts them (VA).
        "held_true_v": float(test["target"][is_held, 0].mean()),
        "held_true_a": float(test["target"][is_held, 1].mean()),
        "held_pred_v": float(test["mean"][is_held, 0].mean()),
        "held_pred_a": float(test["mean"][is_held, 1].mean()),
    }
    if np.any(test["var"] > 0):
        u = np.sqrt(test["var"]).sum(1)
        m["held_auroc"] = auroc(u[seen], u[is_held])
    return m


def reference_coverage(ref_dir: Path, method: str, seed: str, held: int) -> float | None:
    """Joint coverage on expression `held` in the main run, where it was seen."""
    f = ref_dir / f"v2_{method}_s{seed}.npz"
    if not f.exists():
        return None
    npz = np.load(f)
    val, test = load_split(npz, "val"), load_split(npz, "test")
    t = conformal(val, {"test": test})["test"]
    k = test["emotion"] == held
    return float(t["joint"][k].mean()) if k.any() else None


def summarise(runs: dict) -> list[dict]:
    rows = []
    for (method, held), per_seed in sorted(runs.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        row = {"method": method, "held": EMOTIONS[held], "seeds": len(per_seed)}
        keys = sorted({k for m in per_seed for k in m})
        for k in keys:
            vals = [m[k] for m in per_seed if m.get(k) is not None]
            if not vals:
                continue
            row[k] = float(np.mean(vals))
            if len(vals) > 1 and k != "n_held":
                row[k + "_std"] = float(np.std(vals, ddof=1))
        rows.append(row)
    return rows


def latex(rows: list[dict], method: str) -> str:
    pct = lambda r, k: (f"{100 * r[k]:.1f}" + (f"$\\pm${100 * r[k + '_std']:.1f}"
                                              if k + "_std" in r else "")) if k in r else "--"
    num = lambda r, k: f"{r[k]:.2f}" if k in r else "--"
    has_ref = any("held_cov_ref" in r for r in rows if r["method"] == method)
    cols = "lr" + "r" * (6 if has_ref else 5)
    head = ["Held out", "$n$"] + (["In training"] if has_ref else []) + \
           ["Per-axis", "Bonferroni", "Max-score", "Others", "AUROC"]
    out = [f"\\begin{{tabular}}{{{cols}}}", r"\toprule", " & ".join(head) + r" \\", r"\midrule"]
    for r in rows:
        if r["method"] != method:
            continue
        cells = [r["held"], str(int(r["n_held"]))] + ([pct(r, "held_cov_ref")] if has_ref else []) + \
                [pct(r, "held_cov_perdim"), pct(r, "held_cov_bonf"), pct(r, "held_cov_joint"),
                 pct(r, "seen_cov_joint"), num(r, "held_auroc")]
        out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default=None, help="Main-run preds dir (v2_<method>_s<seed>.npz).")
    ap.add_argument("--tables", default=None,
                    help="Also write the evidential table here as loeo.tex (paper_v2/tables).")
    args = ap.parse_args()

    runs: dict = defaultdict(list)
    for f in sorted(Path(args.preds).glob("loeo_*.npz")):
        mt = NAME.search(f.name)
        if not mt:
            continue
        method, held, seed = mt["method"], int(mt["held"]), mt["seed"]
        npz = np.load(f)
        m = evaluate(load_split(npz, "val"), load_split(npz, "test"), held)
        if args.ref:
            m["held_cov_ref"] = reference_coverage(Path(args.ref), method, seed, held)
        runs[(method, held)].append(m)
        print(f"{f.name}: held={EMOTIONS[held]} n={m['n_held']} "
              f"cov_joint={m['held_cov_joint']:.3f} seen={m['seen_cov_joint']:.3f}")
    if not runs:
        sys.exit(f"no loeo_*.npz files in {args.preds}")

    rows = summarise(runs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "loeo.json").write_text(json.dumps(rows, indent=2))
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("method", "held", "seeds"), k))
    with open(out / "loeo.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    for method in sorted({r["method"] for r in rows}):
        (out / f"loeo_{method}.tex").write_text(latex(rows, method))
    if args.tables:
        Path(args.tables).mkdir(parents=True, exist_ok=True)
        (Path(args.tables) / "loeo.tex").write_text(latex(rows, "evidential"))
    print(f"wrote {out}/loeo.json, loeo.csv and one .tex table per method")


if __name__ == "__main__":
    main()

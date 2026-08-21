"""Multi-seed aggregation -> the paper's headline comparison table.

    python -m eval.aggregate_seeds [--root results/summary/real] [--out paper]

Scans <root>/<method>/seed*/metrics.json (one directory per seed, written by
run_eval.py) and emits paper/comparison_table.{csv,tex} with mean +/- std over
seeds. This is the table \input by paper/sections/results.tex, so the paper can
never drift from the artifacts under <root>.

Std is the population standard deviation (ddof=0) over the available seeds.
Methods with a single seed are marked with a double dagger and reported without
error bars. The deterministic baseline emits no uncertainty, so its calibration,
sharpness and OOD columns are degenerate rather than competitive; they are marked
with a dagger and excluded from the "best" comparison.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import pstdev

# (header, dotted path into metrics.json, lower_is_better)
COLUMNS = [
    (r"CCC\,$\uparrow$",       "acc.ccc_mean",          False),
    (r"RMSE\,$\downarrow$",    "acc.rmse_mean",         True),
    (r"ECE\,$\downarrow$",     "ece",                   True),
    (r"Sharp.\ ",              "sharpness",             None),   # None: never bolded
    (r"AURC\,$\downarrow$",    "aurc",                  True),
    (r"OOD-AUROC\,$\uparrow$", "shift.ood_auroc",       False),
    # dECE: all uncertainty-aware methods sit within noise of zero, so ranking
    # them would over-claim a difference the seeds do not support. Never bolded.
    (r"$\Delta$ECE\,$\downarrow$", "shift.ece_degradation", None),
]
CSV_HEADERS = ["CCC", "RMSE", "ECE", "Sharpness", "AURC", "OOD-AUROC", "dECE"]

# Row order and display names.
METHODS = [("baseline", "Baseline"), ("gaussian_nll", "Gaussian NLL"),
           ("evidential", "Evidential"), ("conformal", "Conformal")]

# Columns where the deterministic baseline has no meaningful value.
DEGENERATE = {"ece", "sharpness", "shift.ood_auroc", "shift.ece_degradation"}
# Accuracy columns compare all methods; uncertainty columns exclude the baseline.
ACCURACY_COLS = {"acc.ccc_mean", "acc.rmse_mean"}


def _dig(d: dict, path: str):
    cur = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _abs_delta(path: str, v: float) -> float:
    """dECE is a degradation: rank by magnitude, not signed value."""
    return abs(v) if path == "shift.ece_degradation" else v


def collect(root: Path) -> dict[str, list[dict]]:
    """method -> [metrics dict per seed], ordered by seed number."""
    runs: dict[str, list[dict]] = {}
    for method, _ in METHODS:
        seed_dirs = sorted((root / method).glob("seed*"),
                           key=lambda p: int(p.name[4:]))
        for sd in seed_dirs:
            mj = sd / "metrics.json"
            if mj.exists():
                runs.setdefault(method, []).append(json.loads(mj.read_text()))
    return runs


def _fmt_cell(mean: float, std: float | None, degenerate: bool, best: bool) -> str:
    body = rf"\mathbf{{{mean:.3f}}}" if best else f"{mean:.3f}"
    if degenerate:
        return f"${body}^{{\\dagger}}$"
    if std is None:
        return f"${body}$"
    # Drop the leading zero on the std so the cell stays narrow: 0.017 -> .017
    return rf"${body}_{{\pm{f'{std:.3f}'.lstrip('0')}}}$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/summary/real")
    ap.add_argument("--out", default="paper")
    args = ap.parse_args()

    root = Path(args.root)
    runs = collect(root)
    if not runs:
        raise SystemExit(f"No <method>/seed*/metrics.json found under {root}/.")

    # method -> path -> (mean, std or None)
    stats: dict[str, dict[str, tuple[float, float | None]]] = {}
    for method, seeds in runs.items():
        stats[method] = {}
        for _, path, _ in COLUMNS:
            vals = [v for v in (_dig(m, path) for m in seeds) if v is not None]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            stats[method][path] = (mean, pstdev(vals) if len(vals) > 1 else None)

    # Which cell wins each column.
    best: dict[str, str] = {}
    for _, path, lower in COLUMNS:
        if lower is None:
            continue
        pool = [m for m in stats
                if path in stats[m]
                and not (m == "baseline" and path in DEGENERATE)]
        if path not in ACCURACY_COLS:
            pool = [m for m in pool if m != "baseline"]
        if pool:
            best[path] = min(pool, key=lambda m: _abs_delta(path, stats[m][path][0])
                             * (1 if lower else -1))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- CSV (machine-readable, mean and std as separate columns) ---
    with open(out / "comparison_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Method", "Seeds"] +
                   [c for h in CSV_HEADERS for c in (h, h + "_std")])
        for method, label in METHODS:
            if method not in stats:
                continue
            row = [label, len(runs[method])]
            for (_, path, _) in COLUMNS:
                mean, std = stats[method].get(path, (None, None))
                row += ["" if mean is None else f"{mean:.6f}",
                        "" if std is None else f"{std:.6f}"]
            w.writerow(row)

    # --- LaTeX (the table the paper \inputs) ---
    lines = [r"% GENERATED by eval/aggregate_seeds.py -- do not edit by hand.",
             rf"% Source: {root}/<method>/seed*/metrics.json",
             r"\setlength{\tabcolsep}{4.5pt}",
             r"\begin{tabular}{l" + "c" * len(COLUMNS) + "}",
             r"\toprule",
             " & ".join(["Method"] + [h for h, _, _ in COLUMNS]) + r" \\",
             r"\midrule"]
    for method, label in METHODS:
        if method not in stats:
            continue
        single = len(runs[method]) == 1
        name = label + (r"$^{\ddagger}$" if single else "")
        cells = []
        for (_, path, _) in COLUMNS:
            if path not in stats[method]:
                cells.append("---")
                continue
            mean, std = stats[method][path]
            cells.append(_fmt_cell(mean, std,
                                   degenerate=(method == "baseline" and path in DEGENERATE),
                                   best=(best.get(path) == method)))
        lines.append(" & ".join([name] + cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "comparison_table.tex").write_text("\n".join(lines) + "\n")

    # --- Console preview ---
    hdr = ["Method", "n"] + CSV_HEADERS
    print(" | ".join(h.ljust(12) for h in hdr))
    for method, label in METHODS:
        if method not in stats:
            continue
        cells = [label.ljust(12), str(len(runs[method])).ljust(12)]
        for (_, path, _) in COLUMNS:
            mean, std = stats[method].get(path, (None, None))
            cells.append(("—" if mean is None else
                          f"{mean:.3f}" + (f"±{std:.3f}" if std else "")).ljust(12))
        print(" | ".join(cells))
    print(f"\nWrote {out}/comparison_table.csv and {out}/comparison_table.tex")


if __name__ == "__main__":
    main()

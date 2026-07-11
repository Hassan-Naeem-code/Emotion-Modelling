"""Aggregate every experiment's metrics.json into one comparison table.

    python -m eval.aggregate [--results results] [--out paper]

Scans results/<name>/metrics.json (written by run_eval.py), flattens the headline
metrics into one row per method, and emits a cross-method comparison as CSV and
LaTeX into paper/. This is the table that goes in the paper's results section.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# (column header, dotted path into metrics.json). Order = column order.
COLUMNS = [
    ("Method", "config"),
    ("CCC-V", "acc.ccc_valence"),
    ("CCC-A", "acc.ccc_arousal"),
    ("RMSE", "acc.rmse_mean"),
    ("ECE", "ece"),
    ("Sharpness", "sharpness"),
    ("AURC", "aurc"),
    ("OOD-AUROC", "shift.ood_auroc"),
    ("Cov(test)", "conformal.coverage_test"),
    ("Cov(OOD)", "conformal.coverage_ood"),
]


def _dig(d: dict, path: str):
    cur = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


# Escape characters that are special in LaTeX (method names like "gaussian_nll"
# would otherwise be read as math subscripts and break the build).
def _tex(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("&", r"\&"),
                 ("%", r"\%"), ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def collect(results_dir: Path) -> list[dict]:
    rows = []
    for mj in sorted(results_dir.glob("*/metrics.json")):
        data = json.loads(mj.read_text())
        rows.append({h: _dig(data, p) for h, p in COLUMNS})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper")
    args = ap.parse_args()

    rows = collect(Path(args.results))
    if not rows:
        print(f"No metrics.json found under {args.results}/. Run eval first.")
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    headers = [h for h, _ in COLUMNS]

    csv_path = out / "comparison_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow([_fmt(r[h]) for h in headers])

    tex_path = out / "comparison_table.tex"
    lines = [r"\begin{tabular}{l" + "r" * (len(headers) - 1) + "}", r"\toprule",
             " & ".join(headers) + r" \\", r"\midrule"]
    for r in rows:
        lines.append(" & ".join(_tex(_fmt(r[h])) for h in headers) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex_path.write_text("\n".join(lines))

    # Console preview.
    widths = [max(len(headers[i]), *(len(_fmt(r[headers[i]])) for r in rows))
              for i in range(len(headers))]
    print(" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(" | ".join(_fmt(r[h]).ljust(widths[i]) for i, h in enumerate(headers)))
    print(f"\nWrote {csv_path} and {tex_path}")


if __name__ == "__main__":
    main()

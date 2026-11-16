"""Backbone-robustness table: the main claims, side by side for two encoders.

Reads the analysis.json written by eval.analyze for each backbone and writes
paper_v2/tables/backbone.tex:

    python -m eval.backbone --summaries results/summary/v2 results/summary/v2c \
        --tables paper_v2/tables
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.analyze import _cell

BACKBONES = {"v2": "EfficientNet-B0", "v2c": "ConvNeXt-Tiny"}
HEADS = [("baseline", "Deterministic"), ("gaussian_nll", "Gaussian"),
         ("evidential", "Evidential")]
COLS = ["ccc", "cov_perdim_joint", "cov_joint", "area_joint", "area_bonf",
        "worst_seen_joint", "contempt_cov_joint", "shift_cov_joint"]


def table(summaries: list[Path]) -> str:
    lines = [r"\begin{tabular}{llcccccccc}", r"\toprule",
             r"Backbone & Head & CCC & Per-axis & Joint & Area & Area (Bonf.) & "
             r"Worst & Contempt & Shift \\", r"\midrule"]
    for i, path in enumerate(summaries):
        summary = json.loads((path / "analysis.json").read_text())["summary"]
        prefix = path.name
        if i:
            lines.append(r"\midrule")
        for j, (head, name) in enumerate(HEADS):
            row = summary[f"{prefix}_{head}"]
            label = rf"{BACKBONES[prefix]} ($n={row['n']}$)" if j == 0 else ""
            lines.append(" & ".join([label, name] + [_cell(row, k, digits=3) for k in COLS])
                         + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", nargs="+", required=True,
                    help="analysis dirs named by run prefix (e.g. results/summary/v2)")
    ap.add_argument("--tables", required=True)
    args = ap.parse_args()
    out = Path(args.tables)
    out.mkdir(parents=True, exist_ok=True)
    (out / "backbone.tex").write_text(table([Path(s) for s in args.summaries]))
    print(f"wrote {out / 'backbone.tex'}")


if __name__ == "__main__":
    main()

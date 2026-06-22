# Paper

LaTeX source for the results write-up. It `\input`s artifacts that the pipeline
generates, so the paper always reflects the latest run.

## Generated artifacts (do not edit by hand)
Produced by `make repro` / `eval/run_eval.py` / `eval/aggregate.py`:
- `comparison_table.tex` — cross-method results table (Table 1).
- `<method>_results_table.tex` — per-method table.
- `<method>_risk_coverage.png`, `<method>_reliability.png`,
  `<method>_va_scatter.png` — figures.

## Build
```bash
# from repo root, regenerate numbers first:
make repro && make aggregate
# then build the PDF:
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

If a figure/table is missing, you haven't run the corresponding experiment yet —
run `make repro` (synthetic) or train on real data first. On synthetic data the
numbers validate the machinery; for paper-grade results, train on AffectNet and
evaluate the AffectNet→AFEW-VA shift (`make shift`).

## Hand-written sections
`sections/*.tex` are yours to edit: introduction, method, experiments, results
prose, conclusion. The results section references the generated table/figures by
label.

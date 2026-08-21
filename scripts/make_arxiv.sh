#!/usr/bin/env bash
# Build the arXiv source tarball and verify it the way arXiv will:
# three pdflatex passes, no BibTeX, no refs.bib -- relying on the bundled .bbl.
# Fails loudly rather than shipping a package that will not compile on arXiv.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/arxiv"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cd "$ROOT/paper"
# main.bbl must be current: regenerate it from refs.bib before staging.
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
bibtex main >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true

mkdir -p "$STAGE/pkg/sections"
cp main.tex comparison_table.tex main.bbl "$STAGE/pkg/"
cp evidential_reliability.png evidential_risk_coverage.png evidential_va_scatter.png "$STAGE/pkg/"
cp sections/*.tex "$STAGE/pkg/sections/"

# Clean-room verification (this is the point of the script).
cp -r "$STAGE/pkg" "$STAGE/verify"
cd "$STAGE/verify"
for i in 1 2 3; do
  pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null 2>&1 \
    || { echo "ERROR: arXiv-style build failed on pass $i" >&2; exit 1; }
done
if grep -qE "Warning: (Citation|Reference).*undefined" main.log; then
  echo "ERROR: undefined citations/references in the arXiv build:" >&2
  grep -E "Warning: (Citation|Reference).*undefined" main.log >&2
  exit 1
fi
PAGES=$(grep -o 'Output written on main.pdf ([0-9]* pages' main.log | grep -o '[0-9]*' | head -1)

mkdir -p "$OUT"
cd "$STAGE/pkg"
COPYFILE_DISABLE=1 tar --exclude='.DS_Store' --exclude='._*' \
  -czf "$OUT/arxiv-submission.tar.gz" .
echo "Wrote $OUT/arxiv-submission.tar.gz (verified: ${PAGES} pages, no undefined refs)"

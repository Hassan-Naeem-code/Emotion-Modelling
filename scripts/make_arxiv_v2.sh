#!/usr/bin/env bash
# Build the arXiv source tarball for paper_v2 (replacement of the v1 entry) and
# verify it the way arXiv will: three pdflatex passes, no BibTeX, no refs.bib.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/arxiv/v2"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cd "$ROOT/paper_v2"
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
bibtex main >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true

mkdir -p "$STAGE/pkg/sections" "$STAGE/pkg/tables"
cp main.tex main.bbl "$STAGE/pkg/"
cp sections/*.tex "$STAGE/pkg/sections/"
cp tables/*.tex tables/*.pdf "$STAGE/pkg/tables/"
# The \tbd macro must not survive into a submission.
if grep -rn '\\tbd{' "$STAGE/pkg/sections" >/dev/null; then
  echo "ERROR: unfilled \\tbd{} placeholders remain:" >&2
  grep -rn '\\tbd{' "$STAGE/pkg/sections" >&2; exit 1
fi

cp -r "$STAGE/pkg" "$STAGE/verify"
cd "$STAGE/verify"
for i in 1 2 3; do
  pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null 2>&1 \
    || { echo "ERROR: arXiv-style build failed on pass $i" >&2; exit 1; }
done
if grep -qE "Warning: (Citation|Reference).*undefined" main.log; then
  echo "ERROR: undefined citations/references in the arXiv build:" >&2
  grep -E "Warning: (Citation|Reference).*undefined" main.log >&2; exit 1
fi
PAGES=$(grep -o 'Output written on main.pdf ([0-9]* pages' main.log | grep -o '[0-9]*' | head -1)

mkdir -p "$OUT"
cp main.pdf "$OUT/preview.pdf"
cd "$STAGE/pkg"
COPYFILE_DISABLE=1 tar --exclude='.DS_Store' --exclude='._*' -czf "$OUT/arxiv-v2-submission.tar.gz" .
echo "Wrote $OUT/arxiv-v2-submission.tar.gz (verified: ${PAGES} pages, no undefined refs, no placeholders)"

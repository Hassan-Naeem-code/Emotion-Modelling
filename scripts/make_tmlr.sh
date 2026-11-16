#!/usr/bin/env bash
# Build the TMLR package from the shared paper_v2 sources.
#
#   bash scripts/make_tmlr.sh
#
# Writes:
#   tmlr/submission/paper_anonymous.pdf      -> upload to OpenReview (main PDF)
#   tmlr/submission/supplementary_code.zip   -> upload as supplementary material
#   tmlr/camera_ready/paper_named.pdf        -> named version (after acceptance)
#   tmlr/camera_ready/source_named.zip       -> its LaTeX source
#
# Fails if a build has undefined references, leftover \tbd placeholders, or if
# anything identifying appears in the anonymous PDF or the supplementary zip.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
OUT=$ROOT/tmlr
IDENT='naeem|hassan|va-uncertainty-demo|hassan-naeem\.com|Emotion-Modelling'
rm -rf "$OUT" && mkdir -p "$OUT/submission" "$OUT/camera_ready"

build() { # $1 = anon|named
  local mode=$1 dir=$OUT/build_$1
  mkdir -p "$dir"
  cp paper_tmlr/{main.tex,tmlr.sty,tmlr.bst,fancyhdr.sty} "$dir/"
  cp -R paper_v2/sections paper_v2/tables paper_v2/refs.bib "$dir/"
  [[ $mode == anon ]] && echo '\anontrue' > "$dir/mode.tex" || echo '\anonfalse' > "$dir/mode.tex"
  # Abstract: single source of truth is paper_v2/main.tex.
  python3 - "$dir/abstract.tex" <<'PY'
import re, sys
s = open("paper_v2/main.tex").read()
open(sys.argv[1], "w").write(re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", s, re.S).group(1).strip() + "\n")
PY
  # TMLR is author-year: "Name et al.~\cite{k}" -> \citet{k}; every other \cite -> \citep.
  for f in "$dir"/sections/*.tex; do
    perl -0pi -e 's/[A-Z][A-Za-z-]+\s+et\s+al\.~\\cite\{([^}]+)\}/\\citet{$1}/g; s/\\cite\{/\\citep{/g' "$f"
  done
  if grep -n '\\tbd{' "$dir"/sections/*.tex; then echo "FAIL: \\tbd left"; exit 1; fi
  (cd "$dir" && pdflatex -interaction=nonstopmode main.tex >/dev/null || true
   bibtex main >/dev/null
   pdflatex -interaction=nonstopmode main.tex >/dev/null || true
   pdflatex -interaction=nonstopmode main.tex >/dev/null)
  if grep -q "undefined" "$dir/main.log"; then grep "undefined" "$dir/main.log"; echo "FAIL: undefined refs ($mode)"; exit 1; fi
  echo "$mode: $(grep -o 'Output written.*' "$dir/main.log")"
}

build anon
build named

# The anonymous PDF must not identify the author.
if pdftotext "$OUT/build_anon/main.pdf" - | grep -n -i -E "$IDENT"; then
  echo "FAIL: identifying text in anonymous PDF"; exit 1
fi
if pdfinfo "$OUT/build_anon/main.pdf" | grep -i -E "$IDENT"; then
  echo "FAIL: identifying PDF metadata"; exit 1
fi
cp "$OUT/build_anon/main.pdf" "$OUT/submission/paper_anonymous.pdf"
cp "$OUT/build_named/main.pdf" "$OUT/camera_ready/paper_named.pdf"
(cd "$OUT/build_named" && zip -q -r "$OUT/camera_ready/source_named.zip" main.tex mode.tex abstract.tex main.bbl tmlr.sty tmlr.bst fancyhdr.sty sections tables refs.bib)

# Anonymised code: the working tree's tracked and non-ignored files (no git
# history), minus the paper, the demo (its pages carry the author's name and
# domain) and Colab notes; links scrubbed.
CODE=$OUT/build_code/code
mkdir -p "$CODE"
CODE_PATHS=(eval train models data configs scripts tests export common.py Makefile requirements.txt README.md
  paper_v2/tables results/v2/preds_public results/summary/v2 results/v2c/preds_public results/summary/v2c)
git ls-files -z -c -o --exclude-standard -- "${CODE_PATHS[@]}" | while IFS= read -r -d '' f; do
  [[ -f $f ]] && printf '%s\0' "$f"
done | tar --null -T - -cf - | tar -x -C "$CODE"
[[ -f $CODE/Makefile && -f $CODE/eval/backbone.py && -d $CODE/results/v2c/preds_public ]] || { echo "FAIL: code tree incomplete"; exit 1; }
rm -f "$CODE"/scripts/make_tmlr.sh "$CODE"/scripts/make_arxiv*.sh
grep -rl -i -E "$IDENT" "$CODE" | while read -r f; do
  perl -pi -e 's#https?://github\.com/Hassan-Naeem-code/Emotion-Modelling(\.git)?#<anonymised-repository>#gi; s#https?://(va-uncertainty-demo\.vercel\.app|hassan-naeem\.com)[^\s)"]*#<anonymised-demo>#gi; s#Hassan-Naeem-code/Emotion-Modelling#<anonymised-repository>#gi' "$f"
done
if grep -rn -i -E "$IDENT" "$CODE"; then echo "FAIL: identifying text in supplementary code"; exit 1; fi
(cd "$OUT/build_code" && zip -q -X -r "$OUT/submission/supplementary_code.zip" code)
rm -rf "$OUT/build_code"  # a second copy of tests/ would confuse pytest
echo "supplementary: $(du -h "$OUT/submission/supplementary_code.zip" | cut -f1)"
echo "OK -> $OUT/submission and $OUT/camera_ready"

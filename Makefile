# Reproducibility entrypoints. Every number traces to a config + seed + results file.
PY ?= python
SEED ?= 1337

.PHONY: v2-labels v2-tables v2-verify arxiv-v2 help install test dryrun train-all eval-all ensemble shift aggregate paper-table paper-figures arxiv repro demo-install demo clean

help:
	@echo "make install     - pip install pinned requirements"
	@echo "make dryrun      - end-to-end smoke test on synthetic data (fast)"
	@echo "make train-all   - train baseline, gaussian, evidential on configured data"
	@echo "make eval-all    - evaluate trained models -> results/ + paper/"
	@echo "make repro       - full pipeline: train-all + eval-all + figures/tables"
	@echo "make ensemble    - train K members + evaluate deep ensemble (K=5)"
	@echo "make shift       - train AffectNet, shift-test on AFEW-VA"
	@echo "make aggregate   - merge all metrics.json into one comparison table"
	@echo "make paper-table - regenerate the paper's multi-seed table from results/summary/real/"
	@echo "make paper-figures - copy the committed real-run figures into paper/"
	@echo "make arxiv        - build + verify the arXiv source tarball into arxiv/"
	@echo "make v2-labels   - attach AffectNet labels (from your data/raw copy) to the public predictions"
	@echo "make v2-tables   - recompute every v2 number/table/figure from the committed predictions"
	@echo "make v2-verify   - recompute into a temp dir and diff against the committed tables"
	@echo "make arxiv-v2    - build + verify the paper_v2 arXiv tarball into arxiv/v2/"
	@echo "make test        - run the Python unit tests"
	@echo "make demo-deploy - test, build locally and deploy the demo to Vercel"
	@echo "make demo        - run the in-browser demo (cd demo && npm run dev)"

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest tests/ -q

# Fast smoke test: trains 2 epochs on 100-sample synthetic data and runs eval.
dryrun:
	$(PY) -m train.train --config configs/evidential.yaml --seed $(SEED) --dry-run
	$(PY) -m eval.run_eval --config configs/evidential.yaml \
		--ckpt results/evidential/seed$(SEED)/best.pt --dry-run

train-all:
	$(PY) -m train.train --config configs/baseline.yaml     --seed $(SEED)
	$(PY) -m train.train --config configs/gaussian_nll.yaml  --seed $(SEED)
	$(PY) -m train.train --config configs/evidential.yaml    --seed $(SEED)

eval-all:
	$(PY) -m eval.run_eval --config configs/baseline.yaml     --ckpt results/baseline/seed$(SEED)/best.pt
	$(PY) -m eval.run_eval --config configs/gaussian_nll.yaml  --ckpt results/gaussian_nll/seed$(SEED)/best.pt
	$(PY) -m eval.run_eval --config configs/evidential.yaml    --ckpt results/evidential/seed$(SEED)/best.pt
	$(PY) -m eval.run_eval --config configs/conformal.yaml     --ckpt results/evidential/seed$(SEED)/best.pt

# Deep ensemble: train K members then evaluate epistemic uncertainty.
ensemble:
	bash scripts/train_ensemble.sh $(K) $(if $(DRYRUN),--dry-run,)
	$(PY) -m eval.run_eval --config configs/ensemble_eval.yaml \
		--ckpt results/ensemble/member1/best.pt $(if $(DRYRUN),--dry-run,)
K ?= 5

# Cross-dataset shift: train on AffectNet, shift-test on AFEW-VA.
shift:
	$(PY) -m train.train --config configs/shift_affectnet_to_afew.yaml --seed $(SEED)
	$(PY) -m eval.run_eval --config configs/shift_affectnet_to_afew.yaml \
		--ckpt results/shift_affectnet_to_afew/seed$(SEED)/best.pt --ood-dataset afew_va

# Merge every results/<name>/metrics.json into one comparison table.
aggregate:
	$(PY) -m eval.aggregate

# The paper's headline table: mean +/- std over the seeds committed under
# results/summary/real/. This is what paper/sections/results.tex \inputs.
paper-table:
	$(PY) -m eval.aggregate_seeds

# The paper's three figures come from the seed-1337 evidential checkpoint, which
# is too large to ship in git. The rendered PNGs are committed alongside that
# run's metrics instead, so the paper is reproducible from the repo alone.
# Build the arXiv source tarball: LaTeX sources + .bbl + figures, no .bib
# (arXiv does not run BibTeX). Verified by building in a clean directory.
arxiv: paper-table paper-figures
	@bash scripts/make_arxiv.sh

paper-figures:
	cp results/summary/real/evidential/seed1337/risk_coverage.png paper/evidential_risk_coverage.png
	cp results/summary/real/evidential/seed1337/reliability.png   paper/evidential_reliability.png
	cp results/summary/real/evidential/seed1337/va_scatter.png    paper/evidential_va_scatter.png
	@echo "paper/ figures refreshed from results/summary/real/"

repro: train-all eval-all aggregate
	@echo "Reproduction complete. See results/ and paper/."

demo-install:
	cd demo && npm install

demo:
	cd demo && npm run dev

# Deploy the demo + results explorer. Built locally and uploaded prebuilt: the
# ONNX models are not in git, and Vercel's uploader would apply the repo-level
# `results/*` ignore rule to demo/public/results. Git-triggered builds are off.
demo-deploy:
	cd demo && npm test && vercel build --prod --yes && vercel deploy --prebuilt --prod --yes

clean:
	rm -rf results/*/ paper/*.png paper/*.tex
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

# ---- v2 (resubmission) -------------------------------------------------------
# Every number in paper_v2 is computed from the per-sample predictions committed
# under results/v2/preds (one .npz per model and seed); no GPU is needed.
V2_PREDS ?= results/v2/preds
# The public prediction files carry no AffectNet labels (licence); this rebuilds
# the full files from your own copy of the dataset under data/raw/AffectNetVA.
v2-labels:
	$(PY) -m eval.labels attach --src results/v2/preds_public --dst $(V2_PREDS) --data-root data/raw

v2-tables:
	$(PY) -m eval.analyze --preds $(V2_PREDS) --out results/summary/v2 \
		--ensemble-of gaussian_nll,evidential --tables paper_v2/tables
	@if [ -d results/v2/feats ]; then $(PY) -m eval.novelty --feats results/v2/feats \
		--preds $(V2_PREDS) --out results/summary/v2 --tables paper_v2/tables; \
	else echo "(results/v2/feats absent: novelty table kept as committed; regenerate with eval.dump_feats)"; fi

v2-verify:
	@T=$$(mktemp -d); \
	$(PY) -m eval.analyze --preds $(V2_PREDS) --out $$T/out \
		--ensemble-of gaussian_nll,evidential --tables $$T/tables >/dev/null && \
	for f in main coverage unseen; do \
		diff -q $$T/tables/$$f.tex paper_v2/tables/$$f.tex >/dev/null \
		&& echo "OK   tables/$$f.tex matches the predictions" \
		|| { echo "DIFF tables/$$f.tex"; diff $$T/tables/$$f.tex paper_v2/tables/$$f.tex; exit 1; }; \
	done; rm -rf $$T

arxiv-v2:
	@bash scripts/make_arxiv_v2.sh

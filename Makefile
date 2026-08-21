# Reproducibility entrypoints. Every number traces to a config + seed + results file.
PY ?= python
SEED ?= 1337

.PHONY: help install test dryrun train-all eval-all ensemble shift aggregate paper-table paper-figures arxiv repro demo-install demo clean

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
	@echo "make test        - run the Python unit tests"
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

clean:
	rm -rf results/*/ paper/*.png paper/*.tex
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

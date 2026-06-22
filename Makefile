# Reproducibility entrypoints. Every number traces to a config + seed + results file.
PY ?= python
SEED ?= 1337

.PHONY: help install test dryrun train-all eval-all ensemble shift aggregate repro demo-install demo clean

help:
	@echo "make install     - pip install pinned requirements"
	@echo "make dryrun      - end-to-end smoke test on synthetic data (fast)"
	@echo "make train-all   - train baseline, gaussian, evidential on configured data"
	@echo "make eval-all    - evaluate trained models -> results/ + paper/"
	@echo "make repro       - full pipeline: train-all + eval-all + figures/tables"
	@echo "make ensemble    - train K members + evaluate deep ensemble (K=5)"
	@echo "make shift       - train AffectNet, shift-test on AFEW-VA"
	@echo "make aggregate   - merge all metrics.json into one comparison table"
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

repro: train-all eval-all aggregate
	@echo "Reproduction complete. See results/ and paper/."

demo-install:
	cd demo && npm install

demo:
	cd demo && npm run dev

clean:
	rm -rf results/*/ paper/*.png paper/*.tex
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

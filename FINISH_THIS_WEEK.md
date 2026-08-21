# Status: the paper is backed by real runs end to end

Everything the paper reports now traces to a committed artifact.

## Numbers

`results/summary/real/<method>/seed<n>/metrics.json` holds all ten real
AffectNet runs — baseline / gaussian_nll / evidential at seeds 1337, 7, 21, plus
single-seed conformal. Regenerate the paper's table from them with:

```bash
make paper-table      # -> paper/comparison_table.{tex,csv}
```

If a number in the paper ever disagrees with a run, this command is the
tie-breaker. The table is generated, never hand-edited.

## Figures

The three embedded figures come from the seed-1337 evidential checkpoint. That
checkpoint (18 MB) is too large for git, so the rendered PNGs are committed
beside that run's metrics and copied into `paper/` with:

```bash
make paper-figures
```

They were regenerated on 2026-08-21 from `evidential_real/best.pt` and verified
against the committed metrics before use:

| metric | regenerated run | committed seed 1337 |
|---|---|---|
| ECE | 0.015081904761904791 | 0.015081904761904791 |
| Sharpness | 0.24547742307186127 | 0.24547742307186127 |
| AURC | 0.047210959810 | 0.047210958973 |
| CCC | 0.682616412640 | 0.682616380879 |
| OOD-AUROC | 0.60646 | 0.60646004 |

Agreement to ~8 decimal places; the residual difference is GPU floating-point
ordering, not a change in result.

## Demo

`demo/public/model.onnx` is the real exported evidential model (18 MB, from the
same seed-1337 checkpoint), not the synthetic placeholder. It is gitignored like
all model binaries; regenerate with:

```bash
python -m export.to_onnx --config configs/evidential.yaml \
    --ckpt <path to evidential best.pt>
```

Note: no `--set` is needed — `to_onnx` reads the backbone from the checkpoint's
stored config. `onnxscript` is required on torch >= 2.5 and is now in
`requirements.txt`.

## Known limitations, stated in the paper

- Conformal is single-seed (marked with a double dagger in the table).
- MC-dropout and deep ensembles are implemented but not run, and are excluded
  from the comparison.
- The shift is corruption-based; the cross-dataset AffectNet->AFEW-VA shift needs
  registration-gated data and has not been run. See `REPRODUCE_REAL.md`.

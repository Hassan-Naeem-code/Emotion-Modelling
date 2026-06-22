# Datasets — download instructions

**No dataset is bundled or auto-downloaded.** All require registration and have
licenses that forbid redistribution. Download each yourself, accept its terms,
and place it under the path in your config's `data.root` (default `data/raw/`).

You can validate all plumbing **without any real data** using the synthetic
dataset (`data.name: synthetic`, the default) or the `--dry-run` flag on the real
loaders (loads only 100 samples).

| Dataset | Role | Target | Where to get it |
|---|---|---|---|
| **AffectNet** | Primary training set | Continuous V/A in [-1,1] | http://mohammadmahoor.com/affectnet/ |
| **AffectNet-VA (Kaggle)** | Primary training set — instant download, keeps V/A | Continuous V/A in [-1,1] | `kaggle datasets download jishnusaravanan/affectnetvae` |
| **AFEW-VA** | OOD / distribution-shift test | Per-frame V/A in [-10,10] → rescaled | https://ibug.doc.ic.ac.uk/resources/afew-va-database/ |
| **RAF-DB** | Robustness check (secondary) | Categorical → heuristic VA prototypes | http://www.whdeng.cn/raf/model1.html |
| **DEAP** (optional) | Physiological / rPPG fusion | EEG/peripheral signals | https://www.eecs.qmul.ac.uk/mmv/datasets/deap/ |

## Expected layout

```
data/raw/
  AffectNet/        # *.csv annotation files + image folders (official release)
  AffectNetVA/      # Kaggle 'affectnetvae': {Train,Validation,Test}/{images,valence,arousal,emotion}
  AFEW-VA/          # per-clip folders, each with frame images + JSON annotations
  RAF-DB/           # list_patition_label.txt + aligned/ images
```

### Fastest path (recommended): AffectNet-VA from Kaggle
Most Kaggle AffectNet copies are categorical-only (8 emotion folders) and drop
the valence/arousal values this project needs. `jishnusaravanan/affectnetvae`
keeps the real continuous V/A as per-image `.npy`:
```bash
pip install kaggle           # put your token at ~/.kaggle/kaggle.json
kaggle datasets download jishnusaravanan/affectnetvae -p data/raw --unzip
python -m train.train --config configs/affectnet_va.yaml   # 58k train / 5k test
```

## Notes
- AffectNet uses sentinel `-2` for unannotated V/A; those rows are dropped.
- AFEW-VA valence/arousal are integers in [-10,10]; we divide by 10.
- RAF-DB is categorical. Per project rules, discrete emotions are **not** a
  training target — the VA mapping is a labeled heuristic for robustness only.
- Splits are deterministic (seeded) and saved under `data/splits/`.

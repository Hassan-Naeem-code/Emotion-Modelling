"""RAF-DB loader — secondary face dataset for robustness checks.

RAF-DB is primarily categorical (7 basic emotions). Per the project's hard rule,
discrete emotions are NOT a training target. We map the categorical label to an
approximate VA prototype ONLY for robustness/auxiliary evaluation, clearly
flagged as a derived heuristic — never as a primary continuous target.

Download (registration required): http://www.whdeng.cn/raf/model1.html
Place under <data.root>/RAF-DB/.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .image_va import ImageVADataset, deterministic_split

# Russell circumplex prototypes (valence, arousal) for RAF-DB's 7 labels.
# HEURISTIC mapping for robustness checks only — not a ground-truth VA signal.
_EMOTION_TO_VA = {
    1: (-0.6, 0.7),   # surprise
    2: (-0.8, 0.5),   # fear
    3: (-0.7, 0.2),   # disgust
    4: (0.8, 0.6),    # happy
    5: (-0.7, -0.5),  # sad
    6: (-0.6, 0.8),   # anger
    7: (0.0, 0.0),    # neutral
}


def _read_manifest(root: Path) -> list[tuple[str, float, float]]:
    label_file = root / "list_patition_label.txt"
    if not label_file.exists():
        raise FileNotFoundError(
            f"RAF-DB label file not found at {label_file}. See data/loaders/README.md."
        )
    df = pd.read_csv(label_file, sep=r"\s+", header=None, names=["file", "label"])
    rows = []
    for _, r in df.iterrows():
        v, a = _EMOTION_TO_VA.get(int(r["label"]), (0.0, 0.0))
        rows.append((f"aligned/{Path(r['file']).stem}_aligned.jpg", v, a))
    return rows


def build(cfg: dict, dry_run: bool = False):
    root = Path(cfg["data"]["root"]) / "RAF-DB"
    manifest = _read_manifest(root)
    if dry_run:
        manifest = manifest[:100]
    tr, va, te = deterministic_split(len(manifest), cfg["seed"])
    d = cfg["data"]
    make = lambda idx: ImageVADataset(
        root, [manifest[i] for i in idx], d["image_size"], d["norm_mean"],
        d["norm_std"], align=d.get("align", False)
    )
    return make(tr), make(va), make(te), None

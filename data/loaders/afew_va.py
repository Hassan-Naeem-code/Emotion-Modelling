"""AFEW-VA loader — used as the DISTRIBUTION-SHIFT / OOD test set.

Train on AffectNet images, test on AFEW-VA video frames. This image->video shift
is the spine of the paper: it stresses calibration and is where uncertainty-aware
models should flag degradation while the baseline silently loses coverage.

Download (registration required):
https://ibug.doc.ic.ac.uk/resources/afew-va-database/
Place under <data.root>/AFEW-VA/. The DB provides per-frame JSON with integer
valence/arousal in [-10, 10]; we rescale to [-1, 1].
"""
from __future__ import annotations

import json
from pathlib import Path

from .image_va import ImageVADataset


def _read_manifest(root: Path) -> list[tuple[str, float, float]]:
    rows: list[tuple[str, float, float]] = []
    jsons = sorted(root.rglob("*.json"))
    if not jsons:
        raise FileNotFoundError(
            f"No AFEW-VA annotation JSON found in {root}. See data/loaders/README.md."
        )
    for jp in jsons:
        with open(jp) as f:
            ann = json.load(f)
        clip_dir = jp.parent
        for frame_id, fr in ann.get("frames", {}).items():
            img = clip_dir / f"{frame_id}.png"
            if not img.exists():
                img = clip_dir / f"{frame_id}.jpg"
            if not img.exists():
                continue
            v = float(fr["valence"]) / 10.0   # [-10,10] -> [-1,1]
            a = float(fr["arousal"]) / 10.0
            rows.append((str(img.relative_to(root)), v, a))
    return rows


def build(cfg: dict, dry_run: bool = False):
    root = Path(cfg["data"]["root"]) / "AFEW-VA"
    manifest = _read_manifest(root)
    if dry_run:
        manifest = manifest[:100]
    d = cfg["data"]
    ds = ImageVADataset(root, manifest, d["image_size"], d["norm_mean"],
                        d["norm_std"], align=d.get("align", False))
    # Entire AFEW-VA is treated as a shifted test set. Returned as both `test`
    # and `ood` so eval can use it directly when name == afew_va.
    return None, None, ds, ds

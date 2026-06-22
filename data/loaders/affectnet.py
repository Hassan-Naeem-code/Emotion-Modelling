"""AffectNet loader — PRIMARY training set (images with continuous V/A).

Download (registration required): http://mohammadmahoor.com/affectnet/
Place the extracted data under <data.root>/AffectNet/ with the official CSV(s).
We read the provided annotation columns `valence` and `arousal` in [-1, 1].

This module builds a manifest then defers to ImageVADataset. It does NOT
download anything. Use --dry-run anywhere upstream to validate plumbing on a
100-sample slice without the full dataset.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .image_va import ImageVADataset, deterministic_split


def _read_manifest(root: Path) -> list[tuple[str, float, float]]:
    # AffectNet ships CSVs with columns including subDirectory_filePath, valence,
    # arousal. Adjust the column names here if your distribution differs.
    csvs = sorted(root.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No AffectNet CSV found in {root}. See data/loaders/README.md."
        )
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    path_col = next(c for c in df.columns if "path" in c.lower() or "file" in c.lower())
    df = df.dropna(subset=[path_col, "valence", "arousal"])
    # AffectNet uses sentinel -2 for "no annotation"; drop those rows.
    df = df[(df["valence"] >= -1) & (df["valence"] <= 1)]
    df = df[(df["arousal"] >= -1) & (df["arousal"] <= 1)]
    return list(zip(df[path_col].astype(str), df["valence"].astype(float),
                    df["arousal"].astype(float)))


def build(cfg: dict, dry_run: bool = False):
    root = Path(cfg["data"]["root"]) / "AffectNet"
    manifest = _read_manifest(root)
    if dry_run:
        manifest = manifest[:100]
    tr, va, te = deterministic_split(len(manifest), cfg["seed"])
    d = cfg["data"]
    make = lambda idx: ImageVADataset(
        root, [manifest[i] for i in idx], d["image_size"], d["norm_mean"],
        d["norm_std"], align=d.get("align", False)
    )
    # AffectNet is in-distribution; no native OOD split (use AFEW-VA for shift).
    return make(tr), make(va), make(te), None

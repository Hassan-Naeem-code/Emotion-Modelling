"""Separate AffectNet labels from the released predictions, and re-attach them.

AffectNet's licence does not allow redistributing its annotations, so the public
prediction files (results/v2/preds_public/) hold only model outputs plus image
ids. Anyone with their own copy of the dataset rebuilds the full files with:

    python -m eval.labels attach --src results/v2/preds_public \
        --dst results/v2/preds --data-root data/raw

and then every analysis runs unchanged (make v2-verify / make v2-tables).

    python -m eval.labels strip --src results/v2/preds --dst results/v2/preds_public
creates the public files (maintainers only).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

LABEL_SUFFIXES = ("_target", "_emotion")
# Prediction-file split prefix -> dataset split folder. The corrupted test split
# is the Test split with image corruptions, in the same order.
SPLIT_DIR = {"val": "Validation", "test": "Test", "corrupt": "Test"}


def _ids(split_dir: Path) -> np.ndarray:
    return np.array(sorted(p.stem for p in (split_dir / "images").glob("*.jpg")))


def _labels(split_dir: Path, ids) -> tuple[np.ndarray, np.ndarray]:
    va = np.array([[float(np.load(split_dir / "valence" / f"{i}_val.npy")),
                    float(np.load(split_dir / "arousal" / f"{i}_aro.npy"))] for i in ids],
                  dtype=np.float32)
    emo = []
    for i in ids:
        f = split_dir / "emotion" / f"{i}_exp.npy"
        emo.append(int(np.load(f)) if f.exists() else -1)
    return np.clip(va, -1, 1), np.array(emo)


def strip(src: Path, dst: Path, data_root: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    root = data_root / "AffectNetVA"
    ids = {p: _ids(root / d) for p, d in SPLIT_DIR.items()}
    for f in sorted(src.glob("*.npz")):
        z = np.load(f)
        out = {k: z[k] for k in z.files if not k.endswith(LABEL_SUFFIXES)}
        for p in SPLIT_DIR:
            if f"{p}_mean" in z.files:
                assert len(ids[p]) == len(z[f"{p}_mean"]), (f.name, p)
                out[f"{p}_ids"] = ids[p]
        np.savez_compressed(dst / f.name, **out)
    print(f"stripped labels from {len(list(src.glob('*.npz')))} files -> {dst}")


def attach(src: Path, dst: Path, data_root: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    root = data_root / "AffectNetVA"
    cache: dict[str, tuple] = {}
    for f in sorted(src.glob("*.npz")):
        z = np.load(f)
        out = {k: z[k] for k in z.files}
        for p, d in SPLIT_DIR.items():
            if f"{p}_ids" not in z.files:
                continue
            local = _ids(root / d)
            if not np.array_equal(local, z[f"{p}_ids"]):
                raise SystemExit(f"{f.name}: your {d} split does not match the image ids "
                                 f"the predictions were made on.")
            if d not in cache:
                cache[d] = _labels(root / d, local)
            out[f"{p}_target"], out[f"{p}_emotion"] = cache[d]
        np.savez_compressed(dst / f.name, **out)
    print(f"attached labels to {len(list(src.glob('*.npz')))} files -> {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["strip", "attach"])
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--data-root", default=Path("data/raw"), type=Path)
    a = ap.parse_args()
    (strip if a.cmd == "strip" else attach)(a.src, a.dst, a.data_root)


if __name__ == "__main__":
    main()

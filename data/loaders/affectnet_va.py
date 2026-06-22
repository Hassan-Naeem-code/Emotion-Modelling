"""AffectNet loader for the Kaggle 'affectnetvae' layout, which (unlike most
Kaggle re-uploads) keeps the CONTINUOUS valence/arousal labels we need.

Layout under <data.root>/AffectNetVA/:
    {Train,Validation,Test}/
        images/<id>.jpg
        valence/<id>_val.npy     # scalar in [-1,1]
        arousal/<id>_aro.npy     # scalar in [-1,1]
        emotion/<id>_exp.npy     # categorical (unused; VA is the target)

Kaggle: https://www.kaggle.com/datasets/jishnusaravanan/affectnetvae
Get it with: python -m kaggle datasets download jishnusaravanan/affectnetvae

We use the dataset's own Train/Validation/Test splits directly. Values are the
real AffectNet VA in [-1,1]; the -2 'uncertain/none' sentinel is dropped.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class AffectNetVADataset(Dataset):
    def __init__(self, split_dir: Path, ids: list[str], image_size: int,
                 norm_mean, norm_std, align: bool = False):
        self.split_dir = Path(split_dir)
        self.ids = ids
        self.image_size = image_size
        self.align = align
        self.mean = np.array(norm_mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(norm_std, dtype=np.float32).reshape(3, 1, 1)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        i = self.ids[idx]
        img = Image.open(self.split_dir / "images" / f"{i}.jpg").convert("RGB")
        if self.align:
            from .align import align_face
            img = align_face(img, self.image_size)
        else:
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        arr = (arr - self.mean) / self.std
        val = float(np.load(self.split_dir / "valence" / f"{i}_val.npy"))
        aro = float(np.load(self.split_dir / "arousal" / f"{i}_aro.npy"))
        target = np.clip(np.array([val, aro], dtype=np.float32), -1.0, 1.0)
        return torch.from_numpy(arr), torch.from_numpy(target)


def _ids_for_split(split_dir: Path, dry_run: bool) -> list[str]:
    img_dir = split_dir / "images"
    if not img_dir.exists():
        raise FileNotFoundError(
            f"AffectNetVA images not found at {img_dir}. Download it from Kaggle "
            f"(jishnusaravanan/affectnetvae) and unzip under data/raw/. "
            f"See data/loaders/README.md."
        )
    ids = sorted(p.stem for p in img_dir.glob("*.jpg"))
    return ids[:100] if dry_run else ids


def build(cfg: dict, dry_run: bool = False):
    root = Path(cfg["data"]["root"]) / "AffectNetVA"
    d = cfg["data"]
    mk = lambda name: AffectNetVADataset(
        root / name, _ids_for_split(root / name, dry_run),
        d["image_size"], d["norm_mean"], d["norm_std"], align=d.get("align", False),
    )
    train = mk("Train")
    val = mk("Validation")
    test = mk("Test")
    # AffectNet is in-distribution; AFEW-VA is the shift set (use --ood-dataset).
    return train, val, test, None

"""Generic image-with-VA-annotations dataset, shared by AffectNet / AFEW-VA /
RAF-DB loaders. Each concrete loader is responsible for producing a manifest:
a list of (relative_image_path, valence, arousal) rows. This class handles
loading, face-aware resizing, normalization, and tensor conversion.

We do NOT bundle or auto-download any dataset (all are registration-gated). See
data/loaders/README.md for how to obtain each and where to place it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageVADataset(Dataset):
    def __init__(self, root: str, manifest: list[tuple[str, float, float]],
                 image_size: int, norm_mean, norm_std, align: bool = False):
        self.root = Path(root)
        self.manifest = manifest
        self.image_size = image_size
        self.align = align
        self.mean = np.array(norm_mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(norm_std, dtype=np.float32).reshape(3, 1, 1)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        rel, val, aro = self.manifest[idx]
        img = Image.open(self.root / rel).convert("RGB")
        if self.align:
            from .align import align_face
            img = align_face(img, self.image_size)
        else:
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        arr = (arr - self.mean) / self.std
        target = np.array([val, aro], dtype=np.float32)
        return torch.from_numpy(arr), torch.from_numpy(target)


def deterministic_split(n: int, seed: int, fracs=(0.8, 0.1, 0.1)):
    """Stable index split saved alongside results for reproducibility."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(fracs[0] * n)
    n_val = int(fracs[1] * n)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

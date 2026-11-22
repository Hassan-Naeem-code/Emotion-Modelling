"""AffectNet loader for the Kaggle 'affectnetvae' layout, which (unlike most
Kaggle re-uploads) keeps the CONTINUOUS valence/arousal labels we need.

Layout under <data.root>/AffectNetVA/:
    {Train,Validation,Test}/
        images/<id>.jpg
        valence/<id>_val.npy     # scalar in [-1,1]
        arousal/<id>_aro.npy     # scalar in [-1,1]
        emotion/<id>_exp.npy     # categorical; not a target, used for per-expression
                                 # analysis and data.holdout_expression (LOEO)

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
                 norm_mean, norm_std, align: bool = False, augment: bool = False):
        self.split_dir = Path(split_dir)
        self.ids = ids
        self.image_size = image_size
        self.align = align
        self.augment = _augmenter(image_size) if augment else None
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
        if self.augment is not None:
            img = self.augment(img)
        arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        arr = (arr - self.mean) / self.std
        val = float(np.load(self.split_dir / "valence" / f"{i}_val.npy"))
        aro = float(np.load(self.split_dir / "arousal" / f"{i}_aro.npy"))
        target = np.clip(np.array([val, aro], dtype=np.float32), -1.0, 1.0)
        return torch.from_numpy(arr), torch.from_numpy(target)


def _augmenter(size: int):
    """Train-time augmentation. Every op preserves the VA label: a horizontal
    flip does not change the expressed emotion, and the crop/rotation are small
    enough to keep the whole face. Colour jitter is deliberately mild and there is
    no blur or noise, so the corruption shift used at test time stays unseen.
    """
    from torchvision import transforms as T
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.RandomRotation(10)], p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ])


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


def _expression(split_dir: Path, i: str) -> int:
    f = split_dir / "emotion" / f"{i}_exp.npy"
    return int(np.load(f)) if f.exists() else -1


def _drop_expression(split_dir: Path, ids: list[str], k: int | None) -> list[str]:
    """Remove every image whose AffectNet expression label is k.

    Used for the leave-one-expression-out study: the held-out expression is
    removed from Train (incl. the selection slice) and from Validation (the
    conformal calibration set), but kept in Test, so on Test it plays the same
    role as contempt does in the main experiments: a label never seen in
    training or calibration.
    """
    if k is None or k < 0:
        return ids
    return [i for i in ids if _expression(split_dir, i) != k]


def build(cfg: dict, dry_run: bool = False):
    root = Path(cfg["data"]["root"]) / "AffectNetVA"
    d = cfg["data"]
    augment = bool(cfg.get("train", {}).get("augment", False))
    held = d.get("holdout_expression")
    held = None if held is None else int(held)
    mk = lambda name, ids=None, aug=False: AffectNetVADataset(
        root / name, ids if ids is not None else _ids_for_split(root / name, dry_run),
        d["image_size"], d["norm_mean"], d["norm_std"], align=d.get("align", False),
        augment=aug,
    )
    # Filter before the selection split so the slice is still a fixed fraction
    # of what the model is allowed to see.
    train_ids = _drop_expression(root / "Train", _ids_for_split(root / "Train", dry_run), held)
    select_ids = None
    frac = float(d.get("holdout_select", 0) or 0)
    if frac > 0:
        # Model selection on a slice of Train, so Validation is never seen during
        # training and stays an exchangeable calibration set for conformal.
        # Fixed rng (not the run seed): every seed holds out the same images.
        perm = np.random.default_rng(0).permutation(len(train_ids))
        n_sel = max(1, int(round(frac * len(train_ids))))
        select_ids = [train_ids[i] for i in sorted(perm[:n_sel])]
        train_ids = [train_ids[i] for i in sorted(perm[n_sel:])]
    train = mk("Train", train_ids, aug=augment)
    if select_ids is not None:
        train.select_ds = mk("Train", select_ids)
    val = mk("Validation", _drop_expression(root / "Validation",
                                            _ids_for_split(root / "Validation", dry_run), held))
    test = mk("Test")
    # AffectNet is in-distribution; AFEW-VA is the shift set (use --ood-dataset).
    return train, val, test, None

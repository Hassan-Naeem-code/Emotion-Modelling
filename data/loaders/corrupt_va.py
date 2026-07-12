"""Corruption-based distribution shift for AffectNet-VA.

Wraps the AffectNet *Test* split and applies common image corruptions
(blur / noise / brightness / contrast / JPEG), in the spirit of ImageNet-C.
This gives a genuine in-distribution -> shifted evaluation using the SAME images
and labels, so no second (registration-gated) dataset is required.

Use as the OOD set:
    python -m eval.run_eval --config configs/evidential.yaml \
        --ckpt results/evidential_real/best.pt \
        --set data.name=affectnet_va --ood-dataset affectnet_va_corrupt

Config knobs (under data:):
    corruption: mixed | gaussian_blur | gaussian_noise | brightness | contrast | jpeg
    severity:   1..5   (default 3)
Only PIL + numpy are used, so nothing extra to install.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .affectnet_va import AffectNetVADataset, _ids_for_split

CORRUPTIONS = ["gaussian_blur", "gaussian_noise", "brightness", "contrast", "jpeg"]


def _corrupt(img: Image.Image, kind: str, severity: int, rng: np.random.Generator) -> Image.Image:
    s = max(1, min(5, int(severity)))
    if kind == "gaussian_blur":
        return img.filter(ImageFilter.GaussianBlur(radius=[0.6, 1.2, 2.0, 3.0, 4.0][s - 1]))
    if kind == "gaussian_noise":
        arr = np.asarray(img, dtype=np.float32)
        sigma = [8, 16, 26, 38, 52][s - 1]
        arr = arr + rng.normal(0.0, sigma, arr.shape).astype(np.float32)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if kind == "brightness":
        return ImageEnhance.Brightness(img).enhance([1.3, 1.6, 0.6, 1.9, 0.4][s - 1])
    if kind == "contrast":
        return ImageEnhance.Contrast(img).enhance([0.75, 0.6, 0.45, 0.3, 0.2][s - 1])
    if kind == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=[45, 30, 20, 12, 7][s - 1])
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    return img


class CorruptAffectNetVADataset(AffectNetVADataset):
    def __init__(self, *args, corruption: str = "mixed", severity: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.corruption = corruption
        self.severity = severity

    def __getitem__(self, idx: int):
        i = self.ids[idx]
        img = Image.open(self.split_dir / "images" / f"{i}.jpg").convert("RGB")
        # Deterministic per-image corruption (seeded by index) so runs are reproducible.
        rng = np.random.default_rng(idx)
        kind = self.corruption
        if kind == "mixed":
            kind = CORRUPTIONS[idx % len(CORRUPTIONS)]
        img = _corrupt(img, kind, self.severity, rng)

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
        import torch
        return torch.from_numpy(arr), torch.from_numpy(target)


def build(cfg: dict, dry_run: bool = False):
    root = Path(cfg["data"]["root"]) / "AffectNetVA"
    d = cfg["data"]
    test_dir = root / "Test"
    ds = CorruptAffectNetVADataset(
        test_dir, _ids_for_split(test_dir, dry_run),
        d["image_size"], d["norm_mean"], d["norm_std"], align=d.get("align", False),
        corruption=d.get("corruption", "mixed"), severity=d.get("severity", 3),
    )
    # Only a shift-test split; no train/val/its-own-ood.
    return None, None, ds, None

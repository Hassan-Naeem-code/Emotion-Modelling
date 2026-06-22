"""Synthetic VA dataset — lets the entire pipeline run end-to-end with no real
data, so plumbing, metrics, and figures can be validated deterministically.

It is intentionally a *toy*: a low-dim latent generates both an image-like tensor
and continuous valence/arousal targets, with controllable label noise. An "OOD"
split applies a feature/mean shift so the shift/coverage experiments have
something to detect. Nothing here is meant to produce paper numbers — it exists
to prove the machinery is correct before real datasets are wired in.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticVA(Dataset):
    def __init__(self, n: int, image_size: int, noise: float, seed: int,
                 shift: float = 0.0):
        self.n = n
        self.image_size = image_size
        rng = np.random.default_rng(seed)
        # Latent factors -> VA. Two latents drive valence/arousal; the rest are
        # nuisance. `shift` perturbs the latent distribution to emulate covariate
        # shift between train and the OOD split.
        z = rng.normal(loc=shift, scale=1.0, size=(n, 8)).astype(np.float32)
        # Clean per-sample valence/arousal signals (the thing the image encodes).
        val = np.tanh(0.8 * z[:, 0] - 0.3 * z[:, 1])
        aro = np.tanh(0.7 * z[:, 2] + 0.4 * z[:, 3])
        self.signal = np.stack([val, aro], axis=1).astype(np.float32)
        # Observed targets add aleatoric label noise on top of the clean signal.
        va = self.signal + rng.normal(0, noise, size=self.signal.shape).astype(np.float32)
        self.va = np.clip(va, -1.0, 1.0).astype(np.float32)
        self.z = z
        self._rng = rng

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        # Render a deterministic pseudo-image from the latent so a CNN has signal.
        z = self.z[idx]
        val, aro = self.signal[idx]
        s = self.image_size
        base = np.zeros((3, s, s), dtype=np.float32)
        xs = np.linspace(-1, 1, s, dtype=np.float32)
        gx, gy = np.meshgrid(xs, xs)
        # Encode each axis as a clean, directly-learnable feature: valence scales
        # a horizontal ramp, arousal a vertical ramp. Both are observable with no
        # phase scrambling, so a CNN can recover each axis (high CCC ceiling).
        # A third channel carries seeded nuisance texture for realism.
        base[0] = val * gx
        base[1] = aro * gy
        base[2] = 0.3 * np.sin(4 * (gx + gy) + z[4])
        img = torch.from_numpy(base)
        target = torch.from_numpy(self.va[idx])
        return img, target


def build_synthetic_splits(cfg: dict, dry_run: bool = False):
    """Return (train, val, test, ood) datasets per the synthetic config block."""
    s = cfg["data"]["synthetic"]
    size = cfg["data"]["image_size"]
    noise = s["noise"]
    seed = cfg["seed"]
    if dry_run:
        # Tiny sizes so the smoke test is seconds, not minutes.
        n_tr, n_va, n_te, n_ood = 128, 64, 64, 64
    else:
        n_tr, n_va, n_te, n_ood = s["n_train"], s["n_val"], s["n_test"], s["n_ood"]
    train = SyntheticVA(n_tr, size, noise, seed + 0)
    val = SyntheticVA(n_va, size, noise, seed + 1)
    test = SyntheticVA(n_te, size, noise, seed + 2)
    ood = SyntheticVA(n_ood, size, noise, seed + 3, shift=s["ood_shift"])
    return train, val, test, ood

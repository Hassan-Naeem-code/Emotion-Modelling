"""AFEW-VA loader. Two protocols:

* shift (default): the whole database is one test set, for models trained on
  AffectNet (image -> video shift). Use via --ood-dataset afew_va.
* within (data.afew_protocol: within): AFEW-VA as a second, independent
  benchmark. Clips are split into Train / Validation / Test so a model is
  trained, conformally calibrated and tested on AFEW-VA alone. This is the
  replication of the AffectNet coverage results on a second dataset.

The split is by CLIP, never by frame: consecutive frames of one clip are near
duplicates, so a frame-level split would put copies of test frames into the
calibration set and overstate coverage. Each frame also carries its clip id
(`groups`), so analysis can report clip-averaged coverage as well.

Download (registration required):
https://ibug.doc.ic.ac.uk/resources/afew-va-database/
Place under <data.root>/AFEW-VA/ (one folder per clip, each with <clip>.json and
its frame images). The JSON gives per-frame integer valence/arousal in
[-10, 10]; we rescale to [-1, 1].
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .image_va import ImageVADataset

# Fractions of CLIPS (not frames) for the within-dataset protocol.
SPLIT_FRACS = (0.6, 0.2, 0.2)


def _read_manifest(root: Path) -> list[tuple[str, float, float, str]]:
    """Rows of (relative image path, valence, arousal, clip id)."""
    rows: list[tuple[str, float, float, str]] = []
    jsons = sorted(root.rglob("*.json"))
    if not jsons:
        raise FileNotFoundError(
            f"No AFEW-VA annotation JSON found in {root}. See data/loaders/README.md."
        )
    for jp in jsons:
        with open(jp) as f:
            ann = json.load(f)
        clip_dir = jp.parent
        clip = str(clip_dir.relative_to(root))
        for frame_id, fr in ann.get("frames", {}).items():
            img = clip_dir / f"{frame_id}.png"
            if not img.exists():
                img = clip_dir / f"{frame_id}.jpg"
            if not img.exists():
                continue
            v = float(fr["valence"]) / 10.0   # [-10,10] -> [-1,1]
            a = float(fr["arousal"]) / 10.0
            rows.append((str(img.relative_to(root)), v, a, clip))
    return rows


def _dataset(root: Path, rows, d: dict, augment: bool = False) -> ImageVADataset:
    ds = ImageVADataset(root, [r[:3] for r in rows], d["image_size"], d["norm_mean"],
                        d["norm_std"], align=d.get("align", False), augment=augment)
    clips = sorted({r[3] for r in rows})
    index = {c: i for i, c in enumerate(clips)}
    ds.groups = np.array([index[r[3]] for r in rows], dtype=np.int64)
    ds.group_names = clips
    return ds


def split_clips(clips: list[str], fracs=SPLIT_FRACS, rng_seed: int = 0):
    """Deterministic clip-level split. Fixed rng (not the run seed), so every
    seed and every method sees the same Train / Validation / Test clips."""
    clips = sorted(set(clips))      # unique clips: callers pass one id per frame
    perm = np.random.default_rng(rng_seed).permutation(len(clips))
    n_tr = int(round(fracs[0] * len(clips)))
    n_va = int(round(fracs[1] * len(clips)))
    pick = lambda idx: {clips[i] for i in idx}
    return pick(perm[:n_tr]), pick(perm[n_tr:n_tr + n_va]), pick(perm[n_tr + n_va:])


def build(cfg: dict, dry_run: bool = False):
    d = cfg["data"]
    root = Path(d["root"]) / "AFEW-VA"
    rows = _read_manifest(root)

    if d.get("afew_protocol", "shift") != "within":
        if dry_run:
            rows = rows[:100]
        ds = _dataset(root, rows, d)
        # Entire AFEW-VA as a shifted test set, returned as both test and ood.
        return None, None, ds, ds

    tr, va, te = split_clips([r[3] for r in rows])
    if dry_run:  # keep a few clips per split so every split is non-empty
        tr, va, te = (set(sorted(s)[:3]) for s in (tr, va, te))
    by = lambda keep: [r for r in rows if r[3] in keep]

    # Model selection on a held-out slice of the TRAIN clips (again by clip),
    # so Validation stays untouched for conformal calibration.
    frac = float(d.get("holdout_select", 0) or 0)
    sel: set[str] = set()
    if frac > 0:
        order = sorted(tr)
        perm = np.random.default_rng(1).permutation(len(order))
        sel = {order[i] for i in perm[:max(1, int(round(frac * len(order))))]}
    augment = bool(cfg.get("train", {}).get("augment", False))
    train = _dataset(root, by(tr - sel), d, augment=augment)
    if sel:
        train.select_ds = _dataset(root, by(sel), d)
    return train, _dataset(root, by(va), d), _dataset(root, by(te), d), None

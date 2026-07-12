"""Dataset factory. Returns (train, val, test, ood) datasets for a config.

Any of the returned splits may be None when a dataset doesn't define it (e.g.
AFEW-VA has no train split — it's a shift test set). Callers must handle None.
"""
from __future__ import annotations

from .loaders import affectnet, affectnet_va, afew_va, corrupt_va, raf_db, synthetic


def build_datasets(cfg: dict, dry_run: bool = False):
    name = cfg["data"]["name"]
    if name == "synthetic":
        return synthetic.build_synthetic_splits(cfg, dry_run=dry_run)
    if name == "affectnet":
        return affectnet.build(cfg, dry_run=dry_run)
    if name == "affectnet_va":
        return affectnet_va.build(cfg, dry_run=dry_run)
    if name == "affectnet_va_corrupt":
        return corrupt_va.build(cfg, dry_run=dry_run)
    if name == "afew_va":
        return afew_va.build(cfg, dry_run=dry_run)
    if name == "raf_db":
        return raf_db.build(cfg, dry_run=dry_run)
    raise ValueError(f"Unknown dataset: {name}")

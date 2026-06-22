"""Shared utilities: deterministic seeding + config loading with inheritance.

Every experiment is fully described by (config file + seed). No tunable magic
numbers live in code; they all live in configs/. This module is the single place
configs are parsed so behaviour is identical across train/eval/export.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

CONFIG_DIR = Path(__file__).parent / "configs"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins). Returns a new dict."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    """Load a YAML config, resolving a single optional `_inherit: <file>` chain.

    Inheritance is relative to the configs/ directory so `_inherit: _base.yaml`
    works regardless of the caller's cwd.
    """
    path = Path(path)
    if not path.exists() and (CONFIG_DIR / path).exists():
        path = CONFIG_DIR / path
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("_inherit", None)
    if parent is not None:
        base = load_config(CONFIG_DIR / parent)
        cfg = _deep_merge(base, cfg)
    return cfg


def _coerce(value: str):
    """Parse a CLI override string into a bool/int/float/list where possible."""
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if "," in value:
        return [_coerce(v.strip()) for v in value.split(",")]
    return value


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply dotted-key overrides like 'data.name=affectnet' onto a config.

    Lets any run be reproduced from (config file + seed + the logged --set list)
    without spawning a config file per combination.
    """
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(raw)
    return cfg


def set_seed(seed: int) -> None:
    """Seed every RNG we touch and request deterministic cuDNN where possible."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device() -> str:
    """Pick the best available device. Apple Silicon (mps) is supported."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

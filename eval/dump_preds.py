"""Run a checkpoint once and save raw per-sample predictions for offline analysis.

    python -m eval.dump_preds --ckpt results/affectnet_va_evidential/seed1337/best.pt \
        --out results/preds/evidential_s1337.npz [--wrapper mc_dropout] [--corrupt]

Saves, for the val and test splits (and the corrupted test split with --corrupt):
    <split>_mean, <split>_var, <split>_target, <split>_emotion, and for
    evidential heads <split>_aleatoric / _epistemic / _nu / _alpha / _beta.

Everything downstream (conformal variants, per-emotion coverage, proper scoring
rules) is pure numpy on these files, so new analyses never need the GPU again.
The model's config is read from the checkpoint, so no --set flags are needed.
"""
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import apply_overrides, ensure_dir, get_device, set_seed  # noqa: E402
from data.build import build_datasets  # noqa: E402
from models.fusion import AffectModel  # noqa: E402
from models.wrappers import _enable_dropout  # noqa: E402

EXTRA_KEYS = ("aleatoric", "epistemic", "nu", "alpha", "beta")


def _emotions(ds) -> np.ndarray:
    """AffectNet categorical label per sample (-1 if the dataset has none)."""
    base = getattr(ds, "base", ds)          # corrupted wrapper keeps .base
    split_dir = getattr(base, "split_dir", None)
    ids = getattr(base, "ids", None)
    if split_dir is None or ids is None:
        return np.full(len(ds), -1)
    out = []
    for i in ids:
        f = Path(split_dir) / "emotion" / f"{i}_exp.npy"
        out.append(int(np.load(f)) if f.exists() else -1)
    return np.asarray(out)


@torch.no_grad()
def _collect(model, ds, cfg, device, mc_samples: int = 0) -> dict:
    loader = DataLoader(ds, batch_size=cfg["data"]["batch_size"], shuffle=False,
                        num_workers=cfg["data"]["num_workers"])
    model.eval()
    if mc_samples:
        _enable_dropout(model)
    acc: dict[str, list] = {}
    for x, y in loader:
        x = x.to(device)
        passes = max(1, mc_samples)
        outs = [model(x) for _ in range(passes)]
        means = torch.stack([o["mean"] for o in outs])          # (S,B,2)
        rec = {"mean": means.mean(0), "target": y}
        if "var" in outs[0]:
            rec["var"] = torch.stack([o["var"] for o in outs]).mean(0)
        else:
            rec["var"] = torch.zeros_like(rec["mean"])
        if mc_samples:
            rec["mc_epistemic"] = means.var(0)
            rec["var"] = rec["var"] + rec["mc_epistemic"]
        for k in EXTRA_KEYS:
            if k in outs[0]:
                rec[k] = torch.stack([o[k] for o in outs]).mean(0)
        for k, v in rec.items():
            acc.setdefault(k, []).append(v.detach().cpu().numpy())
    return {k: np.concatenate(v) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mc-samples", type=int, default=0,
                    help=">0 enables MC-dropout with this many passes.")
    ap.add_argument("--corrupt", action="store_true",
                    help="Also dump the corruption-shifted test split.")
    ap.add_argument("--severity", type=int, default=None)
    ap.add_argument("--ood-dataset", default=None,
                    help="Also dump a cross-dataset shift set, e.g. afew_va.")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    args = ap.parse_args()

    device = get_device()
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = apply_overrides(deepcopy(state["cfg"]), args.overrides)
    cfg["model"]["pretrained"] = False      # weights come from the checkpoint
    set_seed(cfg["seed"])
    model = AffectModel(cfg).to(device)
    model.load_state_dict(state["model"])

    _, val_ds, test_ds, _ = build_datasets(cfg)
    splits = {"val": val_ds, "test": test_ds}
    if args.corrupt:
        ccfg = deepcopy(cfg)
        ccfg["data"]["name"] = "affectnet_va_corrupt"
        if args.severity is not None:
            ccfg["data"]["severity"] = args.severity
        _, _, ctest, cood = build_datasets(ccfg)
        splits["corrupt"] = cood if cood is not None else ctest
    if args.ood_dataset:
        ocfg = deepcopy(cfg)
        ocfg["data"]["name"] = args.ood_dataset
        _, _, otest, oood = build_datasets(ocfg)
        splits["xdata"] = oood if oood is not None else otest

    out = {"epoch": np.array(state.get("epoch", -1))}
    for name, ds in splits.items():
        if ds is None:
            continue
        rec = _collect(model, ds, cfg, device, args.mc_samples)
        rec["emotion"] = _emotions(ds)
        groups = getattr(getattr(ds, "base", ds), "groups", None)
        if groups is not None:            # video datasets: clip id per frame
            rec["group"] = np.asarray(groups)
        for k, v in rec.items():
            out[f"{name}_{k}"] = v
        print(f"{name}: {len(ds)} samples")
    ensure_dir(Path(args.out).parent)
    np.savez_compressed(args.out, **out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

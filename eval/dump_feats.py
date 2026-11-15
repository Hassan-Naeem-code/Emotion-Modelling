"""Save penultimate (pooled backbone) features for feature-space novelty detection.

    python -m eval.dump_feats --ckpt results/v2/ckpt/v2_evidential_s1/best.pt \
        --out results/v2/feats/v2_evidential_s1.npz [--n-train 12000]

Saves fp16 features plus emotion labels for: a fixed random subset of the
training split (un-augmented; the reference distribution for the detectors),
val, test, and the corrupted test split. The training subset uses a fixed rng so
every model is compared on the same images.
"""
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ensure_dir, get_device, set_seed  # noqa: E402
from data.build import build_datasets  # noqa: E402
from eval.dump_preds import _emotions  # noqa: E402
from models.fusion import AffectModel  # noqa: E402


@torch.no_grad()
def _feats(model, ds, cfg, device) -> np.ndarray:
    loader = DataLoader(ds, batch_size=cfg["data"]["batch_size"], shuffle=False,
                        num_workers=cfg["data"]["num_workers"])
    model.eval()
    out = [model.backbone(x.to(device)).float().cpu().numpy() for x, _ in loader]
    return np.concatenate(out).astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=12000)
    args = ap.parse_args()

    device = get_device()
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = deepcopy(state["cfg"])
    cfg["model"]["pretrained"] = False
    cfg.setdefault("train", {})["augment"] = False
    set_seed(cfg["seed"])
    model = AffectModel(cfg).to(device)
    model.load_state_dict(state["model"])

    train_ds, val_ds, test_ds, _ = build_datasets(cfg)
    idx = np.sort(np.random.default_rng(0).permutation(len(train_ds))[:args.n_train])
    sub = Subset(train_ds, idx.tolist())
    train_emo = _emotions(train_ds)[idx]

    ccfg = deepcopy(cfg)
    ccfg["data"]["name"] = "affectnet_va_corrupt"
    _, _, ctest, cood = build_datasets(ccfg)
    corrupt_ds = cood if cood is not None else ctest

    out = {}
    for name, ds, emo in [("train", sub, train_emo), ("val", val_ds, None),
                          ("test", test_ds, None), ("corrupt", corrupt_ds, None)]:
        out[f"{name}_feat"] = _feats(model, ds, cfg, device)
        out[f"{name}_emotion"] = emo if emo is not None else _emotions(ds)
        print(f"{name}: {len(ds)} x {out[name + '_feat'].shape[1]}", flush=True)
    ensure_dir(Path(args.out).parent)
    np.savez_compressed(args.out, **out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

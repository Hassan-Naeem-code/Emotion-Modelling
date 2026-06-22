"""Config-driven training entrypoint.

    python -m train.train --config configs/evidential.yaml [--seed N] [--dry-run]

Logs loss curves to results/<name>/seed<seed>/ and saves the best checkpoint by
`train.select_by` (calibration by default, NOT raw MSE — per project spec).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (apply_overrides, ensure_dir, get_device,  # noqa: E402
                    load_config, set_seed)
from data.build import build_datasets  # noqa: E402
from eval.metrics import regression_ece  # noqa: E402
from models.fusion import AffectModel  # noqa: E402
from train.losses import build_loss  # noqa: E402


def _loader(ds, cfg, shuffle):
    return DataLoader(ds, batch_size=cfg["data"]["batch_size"], shuffle=shuffle,
                      num_workers=cfg["data"]["num_workers"], drop_last=False)


@torch.no_grad()
def _evaluate(model, loader, device, loss_fn):
    model.eval()
    means, vars_, targets, losses = [], [], [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        losses.append(loss_fn(out, y.to(device)).item())
        means.append(out["mean"].cpu().numpy())
        vars_.append(out.get("var", torch.zeros_like(out["mean"])).cpu().numpy())
        targets.append(y.numpy())
    mean = np.concatenate(means)
    var = np.concatenate(vars_)
    target = np.concatenate(targets)
    mse = float(np.mean((mean - target) ** 2))
    # Interval calibration error; falls back to MSE when the head has no variance.
    ece = regression_ece(mean, var, target) if var.any() else mse
    return {"loss": float(np.mean(losses)), "mse": mse, "calibration": ece}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="Tiny run to validate plumbing (few epochs, 100 samples).")
    ap.add_argument("--out", default=None, help="Override output dir.")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="Override any config key, e.g. --set data.name=affectnet")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    if args.seed is not None:
        cfg["seed"] = args.seed
    set_seed(cfg["seed"])
    device = get_device()

    if args.dry_run:
        # Smoke test: avoid network weight downloads; random init is fine here.
        cfg["model"]["pretrained"] = False
    epochs = 2 if args.dry_run else cfg["train"]["epochs"]
    out_dir = ensure_dir(args.out or
                         Path(cfg["logging"]["out_dir"]) / cfg["name"] / f"seed{cfg['seed']}")

    try:
        train_ds, val_ds, test_ds, _ = build_datasets(cfg, dry_run=args.dry_run)
    except FileNotFoundError as e:
        sys.exit(f"\n[data not available] {e}\n"
                 f"Dataset '{cfg['data']['name']}' is registration-gated and not "
                 f"bundled. Download it (see data/loaders/README.md) and place it "
                 f"under '{cfg['data']['root']}/', or use the synthetic data with "
                 f"--set data.name=synthetic to test the pipeline without it.")
    if train_ds is None:
        sys.exit(f"\n[no train split] Dataset '{cfg['data']['name']}' has no "
                 f"training split (it's a test/shift-only set). Train on AffectNet "
                 f"or synthetic, and use this dataset via --ood-dataset at eval.")
    train_loader = _loader(train_ds, cfg, shuffle=True)
    val_loader = _loader(val_ds, cfg, shuffle=False)

    model = AffectModel(cfg).to(device)
    loss_fn = build_loss(cfg["model"]["head"], cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])

    select_by = cfg["train"]["select_by"]    # 'calibration' or 'mse'
    best = float("inf")
    history = []
    for ep in range(epochs):
        model.train()
        ep_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            opt.step()
            ep_losses.append(loss.item())

        val = _evaluate(model, val_loader, device, loss_fn)
        score = val[select_by]
        rec = {"epoch": ep, "train_loss": float(np.mean(ep_losses)), **val}
        history.append(rec)
        print(f"[{cfg['name']}] epoch {ep}: train={rec['train_loss']:.4f} "
              f"val_loss={val['loss']:.4f} mse={val['mse']:.4f} "
              f"cal={val['calibration']:.4f} (select_by={select_by})")

        if score < best:
            best = score
            torch.save({"model": model.state_dict(), "cfg": cfg,
                        "epoch": ep, "val": val}, out_dir / "best.pt")

    with open(out_dir / "history.json", "w") as f:
        json.dump({"config": cfg, "history": history, "best": best,
                   "select_by": select_by}, f, indent=2)
    print(f"Saved best ({select_by}={best:.4f}) -> {out_dir/'best.pt'}")


if __name__ == "__main__":
    main()

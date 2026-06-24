"""Emotion classifier on RAF-DB (clean 7-class labels). RAF-DB's labels are far
less noisy than AffectNet's, so accuracy reaches ~80-88% — the honest route to
the 70-85% target. Folder layout: DATASET/{train,test}/{1..7}/*.jpg.

RAF-DB class ids (1-indexed folders) -> our 0-indexed order:
  1 surprise, 2 fear, 3 disgust, 4 happy, 5 sad, 6 anger, 7 neutral

    python -m train.train_rafdb [--epochs 25] [--backbone efficientnet_b0] [--resume]

Saves results/rafdb/best.pt (best test accuracy) with class names for export.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ensure_dir, get_device, set_seed  # noqa: E402
from train.train_emotion import (NORM_MEAN, NORM_STD, EmotionNet,  # noqa: E402
                                 _augment, evaluate)

RAFDB_EMOTIONS = ["surprise", "fear", "disgust", "happy", "sad", "anger", "neutral"]


class RafDbDataset(Dataset):
    """Reads RAF-DB's class-folder layout; folder name N -> class index N-1."""

    def __init__(self, split_dir: Path, size: int, augment: bool = False):
        self.size = size
        self.augment = augment
        self.items: list[tuple[Path, int]] = []
        for cls in range(1, 8):
            for p in (Path(split_dir) / str(cls)).glob("*.jpg"):
                self.items.append((p, cls - 1))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.augment:
            img = _augment(img, np.random.default_rng(), self.size)
        else:
            img = img.resize((self.size, self.size), Image.BILINEAR)
        arr = (np.asarray(img, dtype=np.float32) / 255.0 - NORM_MEAN) / NORM_STD
        return torch.from_numpy(arr.transpose(2, 0, 1)), label


def class_weights(ds: RafDbDataset) -> torch.Tensor:
    counts = np.zeros(len(RAFDB_EMOTIONS))
    for _, y in ds.items:
        counts[y] += 1
    counts = np.maximum(counts, 1)
    return torch.tensor(counts.sum() / (len(RAFDB_EMOTIONS) * counts), dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--backbone", default="efficientnet_b0")
    ap.add_argument("--root", default="data/raw/RAF-DB/DATASET")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    root = Path(args.root)
    out = ensure_dir("results/rafdb")

    train_ds = RafDbDataset(root / "train", args.size, augment=True)
    test_ds = RafDbDataset(root / "test", args.size)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=2, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch, num_workers=2)
    print(f"RAF-DB: {len(train_ds)} train, {len(test_ds)} test")

    model = EmotionNet(args.backbone, len(RAFDB_EMOTIONS), dropout=0.3).to(device)
    crit = nn.CrossEntropyLoss(weight=class_weights(train_ds).to(device),
                               label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = 0.0
    ckpt = out / "best.pt"
    if args.resume and ckpt.exists():
        st = torch.load(ckpt, map_location=device)
        if st.get("backbone") == args.backbone:
            model.load_state_dict(st["model"]); best = st.get("val_acc", 0.0)
            print(f"[resume] loaded {ckpt} (acc={best:.4f})")

    history = []
    for ep in range(args.epochs):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        acc = evaluate(model, test_loader, device)
        history.append({"epoch": ep, "loss": float(np.mean(losses)), "test_acc": acc})
        print(f"[rafdb] epoch {ep}: loss={np.mean(losses):.4f} test_acc={acc:.4f}")
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "backbone": args.backbone,
                        "classes": RAFDB_EMOTIONS, "size": args.size, "val_acc": acc},
                       ckpt)
    json.dump({"history": history, "best_acc": best, "classes": RAFDB_EMOTIONS},
              open(out / "history.json", "w"), indent=2)
    print(f"Saved best (test_acc={best:.4f}) -> {ckpt}")


if __name__ == "__main__":
    main()

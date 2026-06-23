"""Emotion classifier on AffectNet's categorical labels (7 classes), for the
demo's human-readable readout. This complements — does not replace — the
valence/arousal + uncertainty research pipeline: VA stays the scientific target;
this gives the demo a recognizable "Happy / Sad / Angry ..." label that responds
crisply to expressions (a smile reliably reads Happy).

    python -m train.train_emotion [--epochs 10] [--backbone efficientnet_b0]

Saves results/emotion/best.pt (best val accuracy).
"""
from __future__ import annotations

import argparse
import glob
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
from models.backbone import Backbone  # noqa: E402

EMOTIONS = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "anger"]
NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class EmotionDataset(Dataset):
    def __init__(self, split_dir: Path, size: int, ids=None):
        self.d = Path(split_dir)
        self.size = size
        if ids is None:
            ids = sorted(p.stem for p in (self.d / "images").glob("*.jpg"))
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        i = self.ids[idx]
        img = Image.open(self.d / "images" / f"{i}.jpg").convert("RGB").resize(
            (self.size, self.size), Image.BILINEAR)
        arr = (np.asarray(img, dtype=np.float32) / 255.0 - NORM_MEAN) / NORM_STD
        arr = arr.transpose(2, 0, 1)
        label = int(np.load(self.d / "emotion" / f"{i}_exp.npy"))
        return torch.from_numpy(arr), label


class EmotionNet(nn.Module):
    def __init__(self, backbone: str, n_classes: int, dropout: float):
        super().__init__()
        self.backbone = Backbone(backbone, pretrained=True, dropout=dropout)
        self.fc = nn.Linear(self.backbone.out_dim, n_classes)

    def forward(self, x):
        return self.fc(self.backbone(x))


def class_weights(split_dir: Path) -> torch.Tensor:
    files = glob.glob(str(Path(split_dir) / "emotion" / "*.npy"))
    counts = np.zeros(len(EMOTIONS))
    for f in files[:20000]:
        counts[int(np.load(f))] += 1
    counts = np.maximum(counts, 1)
    w = counts.sum() / (len(EMOTIONS) * counts)   # inverse-frequency
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        logits = model(x.to(device))
        pred = logits.argmax(1).cpu()
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--backbone", default="efficientnet_b0")
    ap.add_argument("--root", default="data/raw/AffectNetVA")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    root = Path(args.root)
    out = ensure_dir("results/emotion")

    train_ds = EmotionDataset(root / "Train", args.size)
    val_ds = EmotionDataset(root / "Validation", args.size)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, num_workers=2)

    model = EmotionNet(args.backbone, len(EMOTIONS), dropout=0.2).to(device)
    crit = nn.CrossEntropyLoss(weight=class_weights(root / "Train").to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best = 0.0
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
        acc = evaluate(model, val_loader, device)
        history.append({"epoch": ep, "loss": float(np.mean(losses)), "val_acc": acc})
        print(f"[emotion] epoch {ep}: loss={np.mean(losses):.4f} val_acc={acc:.4f}")
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "backbone": args.backbone,
                        "classes": EMOTIONS, "size": args.size, "val_acc": acc},
                       out / "best.pt")
    json.dump({"history": history, "best_acc": best, "classes": EMOTIONS},
              open(out / "history.json", "w"), indent=2)
    print(f"Saved best (val_acc={best:.4f}) -> {out/'best.pt'}")


if __name__ == "__main__":
    main()

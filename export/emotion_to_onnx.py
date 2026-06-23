"""Export the trained emotion classifier to ONNX for the demo.

    python -m export.emotion_to_onnx --ckpt results/emotion/best.pt \
        --out demo/public/emotion.onnx

Output: `probs` (1, 7) softmax probabilities over
[neutral, happy, sad, surprise, fear, disgust, anger]. Class names + norm stats
are written to demo/public/emotion_meta.json so the demo stays in sync.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.train_emotion import EMOTIONS, NORM_MEAN, NORM_STD, EmotionNet  # noqa: E402


class SoftmaxWrap(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.softmax(self.model(x), dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/emotion/best.pt")
    ap.add_argument("--out", default="demo/public/emotion.onnx")
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cpu")
    model = EmotionNet(state["backbone"], len(state["classes"]), dropout=0.0)
    model.load_state_dict(state["model"])
    model.eval()
    wrap = SoftmaxWrap(model).eval()

    size = state["size"]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrap, torch.randn(1, 3, size, size), str(out_path),
        input_names=["input"], output_names=["probs"],
        dynamic_axes={"input": {0: "batch"}, "probs": {0: "batch"}},
        opset_version=18,
    )
    json.dump({"image_size": size, "classes": state["classes"],
               "norm_mean": NORM_MEAN.tolist(), "norm_std": NORM_STD.tolist(),
               "val_acc": state.get("val_acc")},
              open(out_path.parent / "emotion_meta.json", "w"), indent=2)
    print(f"Exported emotion ONNX -> {out_path}  (val_acc={state.get('val_acc')})")


if __name__ == "__main__":
    main()

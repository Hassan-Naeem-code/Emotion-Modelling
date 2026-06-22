"""Export a trained AffectModel to ONNX for the in-browser demo (ONNX Runtime Web).

    python -m export.to_onnx --config configs/evidential.yaml \
        --ckpt results/evidential/seed1337/best.pt --out demo/public/model.onnx \
        [--conformal results/evidential/conformal.json]

The exported graph takes a normalized face crop (1,3,H,W) and returns:
    va_mean (1,2)   valence, arousal in [-1,1]
    va_std  (1,2)   predictive std per dim (sqrt of total variance)

ONNX Runtime Web was chosen over TF.js for a cleaner PyTorch->ONNX path and to
avoid a TF conversion step. The demo's abstention threshold is read from the
exported conformal JSON (see export below) so the browser uses the SAME
calibrated bound as the paper — uncertainty UX stays faithful to Track A.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import load_config  # noqa: E402
from models.fusion import AffectModel  # noqa: E402


class ExportWrapper(torch.nn.Module):
    """Wraps the model to emit (mean, std) regardless of head type, so the web
    side has a single stable contract. std is 0 for the deterministic baseline."""

    def __init__(self, model: AffectModel):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        mean = out["mean"]
        var = out["var"] if "var" in out else torch.zeros_like(mean)
        std = torch.sqrt(torch.clamp(var, min=1e-8))
        return mean, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="demo/public/model.onnx")
    ap.add_argument("--conformal", default=None,
                    help="Optional conformal.json to copy beside the model for "
                         "the demo's abstention threshold.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = AffectModel(cfg)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()
    wrapper = ExportWrapper(model).eval()

    size = cfg["data"]["image_size"]
    dummy = torch.randn(1, 3, size, size)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, dummy, str(out_path),
        input_names=["input"], output_names=["va_mean", "va_std"],
        dynamic_axes={"input": {0: "batch"}, "va_mean": {0: "batch"},
                      "va_std": {0: "batch"}},
        opset_version=18,
    )

    # Sidecar metadata the demo reads: normalization stats + abstention threshold.
    meta = {
        "image_size": size,
        "norm_mean": cfg["data"]["norm_mean"],
        "norm_std": cfg["data"]["norm_std"],
        "head": cfg["model"]["head"],
        "abstain_std_threshold": 0.35,   # default; overridden by conformal below
    }
    if args.conformal and Path(args.conformal).exists():
        cj = json.loads(Path(args.conformal).read_text())
        # The conformal block may be the top-level object or nested under
        # 'conformal' (run_eval's metrics.json nests it). Handle both.
        conf = cj.get("conformal", cj)
        hw = conf.get("mean_half_width_test")
        if hw is not None:
            meta["abstain_std_threshold"] = float(hw)
        meta["conformal"] = conf
    (out_path.parent / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Exported ONNX -> {out_path}")
    print(f"Wrote sidecar -> {out_path.parent/'model_meta.json'}")


if __name__ == "__main__":
    main()

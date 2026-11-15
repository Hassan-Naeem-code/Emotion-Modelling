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


def joint_calibration(preds_path: str, abstain_q: float, alpha: float = 0.1) -> dict:
    """Max-score conformal quantile and an area threshold, from validation only.
    Also reports, on test, the coverage and abstention rate the demo will have,
    so the shipped numbers are checkable against the paper."""
    import numpy as np
    z = np.load(preds_path)
    sd = lambda p: np.sqrt(np.maximum(z[f"{p}_var"], 1e-12))
    score = lambda p: (np.abs(z[f"{p}_target"] - z[f"{p}_mean"]) / sd(p)).max(1)

    def area(p, q):
        lo = np.clip(z[f"{p}_mean"] - q * sd(p), -1, 1)
        hi = np.clip(z[f"{p}_mean"] + q * sd(p), -1, 1)
        return np.prod(hi - lo, axis=1)

    cal = np.sort(score("val"))
    n = len(cal)
    q = float(cal[min(n - 1, int(np.ceil((n + 1) * (1 - alpha))) - 1)])
    thr = float(np.quantile(area("val", q), abstain_q))
    seen = z["test_emotion"] != 7
    keep = area("test", q) <= thr
    cov = score("test") <= q
    return {
        "joint_q": q,
        "joint_alpha": alpha,
        "abstain_area_threshold": thr,
        "test_check": {
            "joint_coverage_seen": float(cov[seen].mean()),
            "abstain_rate_seen": float((~keep)[seen].mean()),
            "joint_coverage_seen_kept": float(cov[seen & keep].mean()),
            "abstain_rate_contempt": float((~keep)[~seen].mean()),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="demo/public/model.onnx")
    ap.add_argument("--preds", default=None,
                    help="dump_preds .npz of the SAME checkpoint: calibrates the "
                         "joint max-score region (q) and the abstention area on "
                         "its validation split, exactly as in the paper.")
    ap.add_argument("--abstain-quantile", type=float, default=0.9,
                    help="Abstain when the region area exceeds this quantile of "
                         "validation region areas (0.9 -> ~10%% abstention "
                         "on in-distribution faces).")
    ap.add_argument("--conformal", default=None,
                    help="Optional conformal.json to copy beside the model for "
                         "the demo's abstention threshold.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # Prefer the config stored in the checkpoint: it captures the exact training
    # setup (backbone, head, any --set overrides), so export never mismatches the
    # saved weights. Fall back to the YAML config for older checkpoints.
    model_cfg = state.get("cfg", cfg)
    model = AffectModel(model_cfg)
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
    # Inline weights so ONNX Runtime Web (browser) can load a single file —
    # it cannot fetch external .onnx.data sidecars.
    import onnx
    _m = onnx.load(str(out_path))
    onnx.save_model(_m, str(out_path), save_as_external_data=False)
    for _d in out_path.parent.glob(out_path.stem + "*.data"):
        _d.unlink()

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
    if args.preds:
        meta.update(joint_calibration(args.preds, args.abstain_quantile))
    (out_path.parent / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Exported ONNX -> {out_path}")
    print(f"Wrote sidecar -> {out_path.parent/'model_meta.json'}")


if __name__ == "__main__":
    main()

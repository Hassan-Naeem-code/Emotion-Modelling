"""Multimodal model: visual backbone + head, with optional rPPG arousal fusion.

The rPPG branch is quality-gated by design. The scalar rPPG arousal feature and
its quality score are concatenated to the visual features before the head, and
quality multiplicatively gates the feature so a low-quality signal cannot inject
a confident arousal estimate. When rPPG is disabled (the default for image
datasets) the model is purely visual.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import Backbone
from .heads import build_head


class AffectModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        self.use_rppg = m.get("use_rppg", False)
        self.backbone = Backbone(m["backbone"], m["pretrained"], m["dropout"])
        in_dim = self.backbone.out_dim
        if self.use_rppg:
            # +2: gated arousal feature and its quality score.
            in_dim += 2
        self.head = build_head(m["head"], in_dim)
        self.head_kind = m["head"]

    def forward(self, x: torch.Tensor,
                rppg_feature: torch.Tensor | None = None,
                rppg_quality: torch.Tensor | None = None):
        feats = self.backbone(x)
        if self.use_rppg:
            if rppg_feature is None:
                # No signal available -> zero feature, zero quality (max caution).
                rppg_feature = torch.zeros(feats.size(0), device=feats.device)
                rppg_quality = torch.zeros(feats.size(0), device=feats.device)
            gated = (rppg_feature * rppg_quality).unsqueeze(1)
            q = rppg_quality.unsqueeze(1)
            feats = torch.cat([feats, gated, q], dim=1)
        return self.head(feats)


@torch.no_grad()
def predict_with_rppg(model: "AffectModel", image: torch.Tensor,
                      rppg_estimate: dict | None, device: str):
    """Run fused inference for one (or a batch of) face crop(s) given an rPPG
    estimate from models.rppg.estimate(). The quality score gates the feature:
    a low-quality pulse contributes ~0 and the model expresses that as higher
    uncertainty rather than a confident guess. Returns the head output dict.
    """
    model.eval()
    image = image.to(device)
    b = image.size(0)
    if rppg_estimate is None:
        feat = torch.zeros(b, device=device)
        qual = torch.zeros(b, device=device)
    else:
        feat = torch.full((b,), float(rppg_estimate["arousal_feature"]), device=device)
        qual = torch.full((b,), float(rppg_estimate["quality"]), device=device)
    return model(image, rppg_feature=feat, rppg_quality=qual)


def roi_means_from_frames(frames, bbox=None):
    """Mean RGB per frame over the face ROI -> (T,3) array for rppg.estimate().

    frames: iterable of HxWx3 uint8/float arrays. bbox: optional (x,y,w,h) in
    pixels; if None the whole frame is used. This is the video-window front-end
    that turns a clip into the signal rPPG consumes.
    """
    import numpy as np
    out = []
    for fr in frames:
        fr = np.asarray(fr, dtype=np.float32)
        if bbox is not None:
            x, y, w, h = (int(v) for v in bbox)
            fr = fr[y:y + h, x:x + w]
        out.append(fr.reshape(-1, 3).mean(axis=0) if fr.size else np.zeros(3))
    return np.stack(out) if out else np.zeros((0, 3), dtype=np.float32)

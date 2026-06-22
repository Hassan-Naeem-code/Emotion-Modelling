"""Shared visual encoder. Wraps a timm backbone and exposes a flat feature
vector. Swappable via config (model.backbone). Dropout is applied on the
pooled features so MC-Dropout works regardless of the chosen backbone.
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    import timm
except ImportError:  # timm is required for real runs; keep import lazy-friendly
    timm = None


class Backbone(nn.Module):
    def __init__(self, name: str = "resnet50", pretrained: bool = True,
                 dropout: float = 0.2):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required. pip install -r requirements.txt")
        # num_classes=0 + global_pool returns a pooled feature vector.
        self.net = timm.create_model(
            name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        self.out_dim = self.net.num_features
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.net(x)
        return self.dropout(feats)

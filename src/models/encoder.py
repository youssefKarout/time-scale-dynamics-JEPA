"""Shared pair-encoder E_theta.

Consumes a stacked frame pair (X_{t-1}, X_t) -> [B, 6, H, W] and maps it to a
latent state h0 in R^d. GroupNorm is used (instead of BatchNorm) so that the EMA
target copy behaves identically in train/eval without running-stat drift.

Backends (``model.encoder``):
    cnn : the original small convolutional encoder for 96x96 pairs (default).
    vit : ViT-S style encoder on top of a pretrained vision transformer.
          Expects ``timm`` (``pip install timm``). The backbone consumes a
          SINGLE frame (3ch); the pair is encoded by running both frames
          through the shared backbone and fusing with a small head
          (concat | diff | mean). Backbone can be frozen for a linear-probe
          style stage (``vit_freeze=True``) or fully fine-tuned.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _block(in_ch: int, out_ch: int, kernel: int = 3, stride: int = 2) -> nn.Sequential:
    pad = kernel // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False),
        nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
        nn.SiLU(inplace=True),
    )


class CNNEncoder(nn.Module):
    """Small convolutional encoder for 96x96 (or 128x128) RGB frame pairs."""

    def __init__(self, in_channels: int = 6, hidden: int = 256, out_dim: int = 256) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            _block(in_channels, hidden // 4, kernel=5, stride=2),   # H/2
            _block(hidden // 4, hidden // 2),                       # H/4
            _block(hidden // 2, hidden),                            # H/8
            _block(hidden, hidden),                                 # H/16
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 6, H, W] -> h: [B, out_dim]."""
        z = self.stem(x)
        z = self.pool(z).flatten(1)
        return self.proj(z)


class ViTPairEncoder(nn.Module):
    """Pretrained-ViT pair encoder.

    Splits the [B,6,H,W] pair into two [B,3,H,W] frames, encodes each with a
    shared ViT backbone (timm, e.g. ``vit_small_patch16_224`` pretrained on
    ImageNet or DINO), then fuses ``[f_prev, f_curr, f_curr - f_prev]`` through
    a LayerNorm + linear head to ``out_dim``.
    """

    def __init__(
        self,
        out_dim: int = 256,
        backbone: str = "vit_small_patch16_224",
        pretrained: bool = True,
        freeze: bool = False,
        img_size: int = 96,
        fuse: str = "concat_diff",
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError("ViT encoder needs `timm`: pip install timm") from e
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool="token",
        )
        self.feat_dim = int(self.backbone.num_features)
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        self.fuse = fuse
        in_dim = self.feat_dim * 3  # [f_prev, f_curr, diff]
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
        )
        self.img_size = img_size

    def _prep(self, f: torch.Tensor) -> torch.Tensor:
        # ViT wants >= ~96px; bilinear-resize if needed; normalize with ImageNet stats.
        if f.shape[-1] != self.img_size or f.shape[-2] != self.img_size:
            f = torch.nn.functional.interpolate(
                f, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=f.device, dtype=f.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=f.device, dtype=f.dtype).view(1, 3, 1, 1)
        return (f - mean) / std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f0, f1 = x[:, :3], x[:, 3:6]
        z0 = self.backbone(self._prep(f0))
        z1 = self.backbone(self._prep(f1))
        return self.head(torch.cat([z0, z1, z1 - z0], dim=-1))


def build_encoder(cfg) -> nn.Module:
    """Factory honouring ``model.encoder`` (default ``cnn``)."""
    m = cfg.model
    kind = str(getattr(m, "encoder", "cnn")).lower()
    d = int(m.latent_dim)
    if kind == "cnn":
        return CNNEncoder(in_channels=6, hidden=int(m.encoder_hidden), out_dim=d)
    if kind == "vit":
        return ViTPairEncoder(
            out_dim=d,
            backbone=str(getattr(m, "vit_backbone", "vit_small_patch16_224")),
            pretrained=bool(getattr(m, "vit_pretrained", True)),
            freeze=bool(getattr(m, "vit_freeze", False)),
            img_size=int(getattr(m, "vit_img_size", 224)),
        )
    raise ValueError(f"unknown model.encoder {kind!r} (cnn|vit)")


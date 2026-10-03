"""Frame augmentations for LTS-D-JEPA.

Design notes:
  * Spatial crops are sampled ONCE per sample and applied to ALL frames of that
    sample (online pair + every target pair) so that geometry/motion direction is
    preserved across time. Colour jitter is sampled independently for the online
    view and the target view (JEPA-style asymmetry).
  * NO horizontal / temporal flips: they would reverse motion direction and
    corrupt the dynamics the velocity fields are meant to model.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class CropParams:
    top: int
    left: int
    height: int
    width: int
    out_height: int
    out_width: int


def sample_crop_params(
    h: int,
    w: int,
    out_h: int,
    out_w: int,
    scale: Tuple[float, float] = (0.6, 1.0),
    ratio: Tuple[float, float] = (0.9, 1.111),
    rng: random.Random | None = None,
) -> CropParams:
    rng = rng or random
    area = h * w
    for _ in range(10):
        target_area = rng.uniform(*scale) * area
        aspect = math.exp(rng.uniform(math.log(ratio[0]), math.log(ratio[1])))
        ch = int(round(math.sqrt(target_area / aspect)))
        cw = int(round(math.sqrt(target_area * aspect)))
        if 2 <= ch <= h and 2 <= cw <= w:
            top = rng.randint(0, h - ch)
            left = rng.randint(0, w - cw)
            return CropParams(top, left, ch, cw, out_h, out_w)
    # fallback: centre crop covering full frame
    return CropParams(0, 0, h, w, out_h, out_w)


def apply_crop(frames: torch.Tensor, crop: CropParams) -> torch.Tensor:
    """frames: [..., 3, H, W] (float). Resizes the crop to (out_h, out_w)."""
    out = frames[..., crop.top : crop.top + crop.height, crop.left : crop.left + crop.width]
    if (crop.height, crop.width) != (crop.out_height, crop.out_width):
        lead = out.shape[:-3]
        flat = out.reshape(-1, out.shape[-3], out.shape[-2], out.shape[-1])
        flat = F.interpolate(flat, size=(crop.out_height, crop.out_width), mode="bilinear", align_corners=False)
        out = flat.reshape(*lead, 3, crop.out_height, crop.out_width)
    return out


@dataclass
class ColorParams:
    brightness: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0


def sample_color_params(
    brightness: float = 0.4,
    contrast: float = 0.4,
    saturation: float = 0.4,
    rng: random.Random | None = None,
) -> ColorParams:
    rng = rng or random

    def _s(v: float) -> float:
        return 1.0 + rng.uniform(-v, v)

    return ColorParams(_s(brightness), _s(contrast), _s(saturation))


def apply_color(frames: torch.Tensor, p: ColorParams) -> torch.Tensor:
    """frames: [..., 3, H, W] in [0, 1]."""
    x = frames
    if p.brightness != 1.0:
        x = x * p.brightness
    if p.saturation != 1.0:
        grey = x.mean(dim=-3, keepdim=True)
        x = grey + (x - grey) * p.saturation
    if p.contrast != 1.0:
        mean = x.mean(dim=(-3, -2, -1), keepdim=True)
        x = mean + (x - mean) * p.contrast
    return x.clamp_(0.0, 1.0)


def to_float(frames_u8: torch.Tensor) -> torch.Tensor:
    return frames_u8.float().div_(255.0)


def normalize(frames: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    m = torch.tensor(mean, dtype=frames.dtype, device=frames.device).view(3, 1, 1)
    s = torch.tensor(std, dtype=frames.dtype, device=frames.device).view(3, 1, 1)
    return (frames - m) / s


def augment_sample(
    online: torch.Tensor,
    targets: List[torch.Tensor],
    out_h: int,
    out_w: int,
    rng: random.Random | None = None,
    color_strength: float = 0.4,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Augment one sample.

    online : [2, 3, H, W] uint8 tensor   (X_{t-1}, X_t)
    targets: list of [2, 3, H, W] uint8  (one per horizon)

    Returns float tensors in normalised space, geometry-consistent across frames.
    """
    online_f = to_float(online)
    targets_f = [to_float(t) for t in targets]

    h, w = online_f.shape[-2:]
    crop = sample_crop_params(h, w, out_h, out_w, rng=rng)

    online_f = apply_crop(online_f, crop)
    targets_f = [apply_crop(t, crop) for t in targets_f]

    online_f = apply_color(online_f, sample_color_params(color_strength, color_strength, color_strength, rng))
    targets_f = [apply_color(t, sample_color_params(color_strength, color_strength, color_strength, rng)) for t in targets_f]

    online_f = normalize(online_f)
    targets_f = [normalize(t) for t in targets_f]
    return online_f, targets_f


def stack_pair_to_channels(pair: torch.Tensor) -> torch.Tensor:
    """[B, 2, 3, H, W] -> [B, 6, H, W]."""
    b, n, c, h, w = pair.shape
    assert n == 2, f"expected 2 frames, got {n}"
    return pair.reshape(b, n * c, h, w)

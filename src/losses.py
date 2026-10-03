"""Losses: JEPA distance + VICReg variance/covariance (anti-collapse only)."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def jepa_distance(pred: torch.Tensor, target: torch.Tensor, kind: str = "cosine") -> torch.Tensor:
    """Distance between predicted and target latents (mean over the batch)."""
    if kind == "cosine":
        return (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
    if kind == "mse":
        return F.mse_loss(pred, target)
    if kind == "smooth_l1":
        return F.smooth_l1_loss(pred, target)
    raise ValueError(f"unknown distance {kind!r}")


def vicreg_terms(x: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (variance_term, covariance_term) for a [B, d] embedding.

    The covariance is computed on features normalised per-dimension to a safe
    unit scale (floor = quarter of the mean std). This keeps the term O(1) even
    though our latents are unnormalised and have heterogeneous per-dim scale.
    """
    n, d = x.shape
    z = x - x.mean(dim=0, keepdim=True)

    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    var_term = torch.mean(F.relu(gamma - std))

    if n <= 1:
        return var_term, torch.zeros((), device=x.device)

    gstd = std.mean().clamp_min(eps)
    scale = std.clamp_min(0.25 * gstd)
    zn = z / scale
    cov = (zn.t() @ zn) / (n - 1)                # per-dim ~unit scale
    off = cov - torch.diag_embed(torch.diagonal(cov))
    cov_term = off.pow(2).sum() / d

    return var_term, cov_term


def vicreg_loss(x: torch.Tensor, var_w: float, cov_w: float, gamma: float, eps: float):
    var_term, cov_term = vicreg_terms(x, gamma=gamma, eps=eps)
    total = var_w * var_term + cov_w * cov_term
    return total, var_term, cov_term

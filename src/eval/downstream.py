"""Downstream evaluation from FROZEN latent states (labels used here only).

Probes h0 (and, optionally, the integrated states) against MOVi-A ground-truth
object trajectories. The GT is never used during training.

GT format (produced by scripts/export_gt_trajectories.py):
    gt/{i:05d}.npz with arrays:
        positions : [T, K, 3] float32
        velocities: [T, K, 3] float32
        object_types: [T, K] int32
Clip index i must align with the video file order used by the dataset.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..data.transforms import normalize, stack_pair_to_channels, to_float


@torch.no_grad()
def extract_h0(model, frames_u8: torch.Tensor, size: int, device: str) -> torch.Tensor:
    """frames_u8: [N,2,3,H,W] uint8 -> h0 [N, d]."""
    x = to_float(frames_u8)
    if x.shape[-1] != size:
        x = torch.nn.functional.interpolate(
            x.reshape(-1, 3, x.shape[-2], x.shape[-1]), size=(size, size), mode="bilinear", align_corners=False
        )
        x = x.reshape(frames_u8.shape[0], 2, 3, size, size)
    x = stack_pair_to_channels(normalize(x)).to(device)
    return model.encode_online(x)


def ridge_fit(X: torch.Tensor, Y: torch.Tensor, alpha: float = 1e-2):
    """Closed-form ridge regression (with bias). Returns (W, b) and reports R^2."""
    n = X.shape[0]
    Xb = torch.cat([X, torch.ones(n, 1, device=X.device, dtype=X.dtype)], dim=1)
    d = Xb.shape[1]
    A = Xb.t() @ Xb + alpha * torch.eye(d, device=X.device, dtype=X.dtype)
    W = torch.linalg.solve(A, Xb.t() @ Y)              # [d+1, out]
    pred = Xb @ W
    ss_res = ((Y - pred) ** 2).sum()
    ss_tot = ((Y - Y.mean(0, keepdim=True)) ** 2).sum()
    r2 = (1.0 - ss_res / (ss_tot + 1e-8)).item()
    return W, r2


def collect_dataset(
    model, dataset, gt_dir: str, t: int, device: str, size: int,
    indices: Optional[List[int]] = None, target_kind: str = "positions",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (h0, y_now, y_future_max_tau) arrays for a fixed start frame t."""
    if indices is None:
        indices = list(range(len(dataset)))
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.npz")))

    feats, y_now, y_fut = [], [], []
    for i in indices:
        clip = dataset.load_clip(i)
        pair = np.stack([clip[t - 1], clip[t]], 0)                 # [2,H,W,3]
        h0 = _encode(model, pair, size, device)
        gt = np.load(gt_files[i])
        pos = gt[target_kind].reshape(gt[target_kind].shape[0], -1)
        feats.append(h0.squeeze(0).cpu())
        y_now.append(torch.from_numpy(pos[t].astype(np.float32)))
        y_fut.append(torch.from_numpy(pos[min(t + max(model.taus), pos.shape[0] - 1)].astype(np.float32)))

    return torch.stack(feats), torch.stack(y_now), torch.stack(y_fut)


def _encode(model, pair: np.ndarray, size: int, device: str) -> torch.Tensor:
    """pair: [2, H, W, 3] uint8 -> h0 [1, d]."""
    x = torch.from_numpy(pair).permute(0, 3, 1, 2).float() / 255.0   # [2,3,H,W]
    x = x.unsqueeze(0)                                              # [1,2,3,H,W]
    x = torch.nn.functional.interpolate(
        x.reshape(-1, 3, x.shape[-2], x.shape[-1]), size=(size, size), mode="bilinear", align_corners=False
    ).reshape(1, 2, 3, size, size)
    x = stack_pair_to_channels(normalize(x)).to(device)
    return model.encode_online(x)


def report_linear_probe(model, dataset, gt_dir: str, t: int, device: str, size: int,
                        n: int = 128, alpha: float = 1e-2) -> Dict[str, float]:
    idx = list(range(min(n, len(dataset))))
    X, y_now, y_fut = collect_dataset(model, dataset, gt_dir, t, device, size, idx)
    _, r2_now = ridge_fit(X, y_now, alpha)
    _, r2_fut = ridge_fit(X, y_fut, alpha)
    return {"probe/r2_pos_now": r2_now, "probe/r2_pos_future": r2_fut}

"""Mechanism M5: per-branch target standardisation with running statistics.

Rationale (hygiene, NOT a scale mechanism): the two branches share the target
encoder, so raw target magnitudes can differ simply because one horizon is
farther. Removing that trivial-magnitude confound means any residual
inter-branch difference must be a *dynamical* difference.

We centre by a per-dimension running mean (safe: subtraction) and scale by a
single per-branch SCALAR (not per-dim division), which equalises branch
magnitudes without exploding on low-variance dimensions. The same fixed,
stop-grad transform is applied to predictions and targets.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TargetStandardizer(nn.Module):
    def __init__(self, n_branches: int, dim: int, momentum: float = 0.99, eps: float = 1e-3) -> None:
        super().__init__()
        self.n = n_branches
        self.dim = dim
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("mean", torch.zeros(n_branches, dim))
        self.register_buffer("scale", torch.ones(n_branches))
        self.register_buffer("initialized", torch.zeros(n_branches))

    @torch.no_grad()
    def update(self, targets) -> None:
        for k, t in enumerate(targets):
            t = t.detach()
            b_mean = t.mean(dim=0)
            b_scale = t.var(dim=0, unbiased=False).mean().sqrt().clamp_min(self.eps)
            if self.initialized[k] < 0.5:
                self.mean[k].copy_(b_mean)
                self.scale[k].copy_(b_scale)
                self.initialized[k] = 1.0
            else:
                m = self.momentum
                self.mean[k].mul_(m).add_(b_mean, alpha=1 - m)
                self.scale[k].mul_(m).add_(b_scale, alpha=1 - m)

    def forward(self, preds, targets):
        out_p, out_t = [], []
        for k in range(self.n):
            mu = self.mean[k].detach()
            sc = self.scale[k].detach().clamp_min(self.eps)
            out_p.append((preds[k] - mu) / sc)
            out_t.append((targets[k] - mu) / sc)
        return out_p, out_t

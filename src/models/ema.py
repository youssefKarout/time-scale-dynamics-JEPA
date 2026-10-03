"""EMA target-encoder update: xi <- beta * xi + (1 - beta) * theta."""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


def copy_params(src: nn.Module, dst: nn.Module) -> None:
    """Hard-copy parameters and buffers (used to initialise the EMA target)."""
    dst.load_state_dict(src.state_dict())


@torch.no_grad()
def ema_update(online: nn.Module, target: nn.Module, beta: float) -> None:
    for po, pt in zip(online.parameters(), target.parameters()):
        pt.mul_(beta).add_(po.detach(), alpha=1.0 - beta)
    for bo, bt in zip(online.buffers(), target.buffers()):
        bt.copy_(bo)

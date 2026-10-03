"""Reproducibility helpers."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (if available)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch not installed yet (e.g. pure-numpy tests)
        pass


def resolve_device(requested: str = "auto") -> str:
    """Resolve 'auto' to cuda/cpu without importing torch at module import time."""
    import torch

    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"

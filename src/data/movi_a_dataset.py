"""MOVi-A RGB dataset.

The PoC stores one .npz per clip with a single array `video` of shape
[T, H, W, 3] uint8 (T = 24 for MOVi-A 128x128). Ground-truth trajectories are
NOT stored here; they are only needed downstream (eval) — see src/eval.

Sampling a training window needs four frame indices for a horizon tau:
    (t-1, t) and (t+tau-1, t+tau).
For T=24 and tau_max=16 this requires t in [1, T-1-tau_max] = [1, 7].
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


class MoviADataset(Dataset):
    def __init__(
        self,
        root: str,
        horizons: List[int],
        num_clips: int | None = None,
        max_start_jitter: int = 0,
        train: bool = True,
        split_seed: int = 0,
        val_fraction: float = 0.0,
    ) -> None:
        self.root = root
        self.horizons = list(horizons)
        self.max_tau = max(self.horizons)
        self.max_start_jitter = int(max_start_jitter)
        self.train = train

        files = sorted(glob.glob(os.path.join(root, "*.npz")))
        if num_clips is not None:
            files = files[:num_clips]
        if not files:
            raise FileNotFoundError(f"No .npz clips found under {root!r}")

        if val_fraction > 0.0:
            n_val = max(1, int(len(files) * val_fraction))
            rng = np.random.RandomState(split_seed)
            perm = rng.permutation(len(files))
            val_idx = set(perm[:n_val].tolist())
            files = [f for i, f in enumerate(files) if (i in val_idx) == (not train)]

        self.files = files

        # Infer T from the first clip.
        with np.load(self.files[0]) as fh:
            self.T = int(fh["video"].shape[0])

        self.t_min = 1
        self.t_max = self.T - 1 - self.max_tau
        if self.t_max < self.t_min:
            raise ValueError(
                f"Clip length T={self.T} too short for horizons {self.horizons}"
            )

    def __len__(self) -> int:
        return len(self.files)

    def load_clip(self, index: int) -> np.ndarray:
        with np.load(self.files[index]) as fh:
            return fh["video"]

    def _choose_start(self, rng: np.random.RandomState) -> int:
        if not self.train or self.max_start_jitter == 0:
            # stratified: cycle through the valid start range deterministically
            span = self.t_max - self.t_min + 1
            return self.t_min + (rng.randint(0, span) if self.train else span // 2)
        jitter = rng.randint(-self.max_start_jitter, self.max_start_jitter + 1)
        return int(np.clip(self.t_min + (self.t_max - self.t_min) // 2 + jitter, self.t_min, self.t_max))

    def __getitem__(self, index: int) -> Dict[str, object]:
        rng = np.random.RandomState((index + (0 if self.train else 7777)) % (2**31 - 1))
        video = self.load_clip(index)
        t = self._choose_start(rng)

        # channel-first pairs: [2, 3, H, W]
        online = np.stack([video[t - 1], video[t]], axis=0).transpose(0, 3, 1, 2)

        targets = []
        for tau in self.horizons:
            pair = np.stack([video[t + tau - 1], video[t + tau]], axis=0).transpose(0, 3, 1, 2)
            targets.append(pair)

        return {
            "online": torch.from_numpy(np.ascontiguousarray(online)),          # [2,3,H,W] uint8
            "targets": [torch.from_numpy(np.ascontiguousarray(a)) for a in targets],
            "t": t,
            "index": index,
        }

    def future_pairs(self, index: int, t: int, taus: List[int]) -> List[np.ndarray]:
        """Return future pairs (X_{s+tau-1}, X_{s+tau}) as [2,3,H,W] for analysis."""
        video = self.load_clip(index)
        out = []
        for tau in taus:
            out.append(np.stack([video[t + tau - 1], video[t + tau]], axis=0).transpose(0, 3, 1, 2))
        return out


def collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    online = torch.stack([b["online"] for b in batch], dim=0)              # [B,2,3,H,W]
    n_tau = len(batch[0]["targets"])
    targets = [
        torch.stack([b["targets"][k] for b in batch], dim=0)              # [B,2,3,H,W]
        for k in range(n_tau)
    ]
    return {
        "online": online,
        "targets": targets,
        "t": torch.tensor([b["t"] for b in batch], dtype=torch.long),
        "index": torch.tensor([b["index"] for b in batch], dtype=torch.long),
    }

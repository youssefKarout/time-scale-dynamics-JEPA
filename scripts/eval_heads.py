"""Evaluate head-based controls (B2 horizon_cond, B3 regressor) on trained + held-out horizons.

Usage:
    python scripts/eval_heads.py --ckpt runs/movi_a_poc/horizon_cond/ckpt_epoch99.pt --n-clips 128
    python scripts/eval_heads.py --ckpt runs/movi_a_poc/regressor/ckpt_epoch99.pt --n-clips 128

Prints the per-tau error table (E2b extrapolation) + head-fit matrix (CHTG analogue).
Reuses the dataset future_pairs() helper, so no loader changes are needed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.movi_a_dataset import MoviADataset  # noqa: E402
from src.metrics.specialization import preprocess_pairs  # noqa: E402
from src.models.lts_dynamics_jepa import LatentTimeScaleJEPA  # noqa: E402
from src.utils.config import to_dotdict  # noqa: E402
from src.losses import jepa_distance  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-clips", type=int, default=128)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = to_dotdict(ckpt["cfg"])
    cfg.metrics.band_horizons = [1, 2, 3, 4, 6, 8, 12, 16, 20]
    model = LatentTimeScaleJEPA(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = MoviADataset(root=cfg.data.root, horizons=list(cfg.data.horizons),
                      num_clips=cfg.data.num_clips, max_start_jitter=0,
                      train=False, val_fraction=cfg.data.val_fraction)
    taus = list(cfg.data.horizons)
    grid = list(cfg.metrics.band_horizons)
    size = int(cfg.data.frame_size)
    n = min(args.n_clips, len(ds))
    idx = list(range(n))
    loader = DataLoader([ds[i] for i in idx], batch_size=64,
                        collate_fn=lambda b: b, shuffle=False)

    fit_acc = np.zeros((2, 2))
    band_acc = [[[] for _ in range(2)] for _ in grid]
    with torch.no_grad():
        for items in loader:
            ids = torch.tensor([b["index"] for b in items])
            ts = torch.tensor([b["t"] for b in items])
            on = torch.stack([b["online"] for b in items])
            h0 = model.encode_online(preprocess_pairs(on, size))
            tg = [preprocess_pairs(torch.stack([b["targets"][k] for b in items]), size)
                  for k in range(2)]
            tgt = [model.encode_target(t) for t in tg]
            for i in range(2):
                for j in range(2):
                    p = model.predict(h0, 0, tau=float(taus[j])) \
                        if model.variant == "horizon_cond" else model.predictors[i](h0)
                    fit_acc[i, j] += jepa_distance(p, tgt[j], kind="cosine").item() * len(items)
            for gi, tau in enumerate(grid):
                fut, valid_idx = [], []
                for bi, b in enumerate(items):
                    if int(b["t"]) + int(tau) <= ds.T - 1:
                        fp = ds.future_pairs(int(b["index"]), int(b["t"]), [int(tau)])[0]
                        fut.append(torch.from_numpy(np.ascontiguousarray(fp)))
                        valid_idx.append(bi)
                if not fut:
                    continue
                tf = model.encode_target(preprocess_pairs(torch.stack(fut), size))
                h0v = h0[torch.tensor(valid_idx)]
                for head in range(2):
                    if model.variant == "horizon_cond":
                        p = model.predict(h0v, 0, tau=float(tau))
                    elif int(tau) == int(taus[head]):
                        p = model.predictors[head](h0v)
                    else:
                        continue
                    band_acc[gi][head].append(jepa_distance(p, tf, kind="cosine").item())

    fit_acc /= n
    diag = float(np.mean(np.diag(fit_acc)))
    off = float((fit_acc.sum() - np.trace(fit_acc)) / 2)
    print(f"=== {args.ckpt}  (variant={cfg.run.variant}, n_clips={n}) ===")
    print(f"  head-fit/diag {diag:+.4f}  offdiag {off:+.4f}  delta {off - diag:+.4f}")
    print(f"  matrix: e00={fit_acc[0,0]:.4f} e01={fit_acc[0,1]:.4f} "
          f"e10={fit_acc[1,0]:.4f} e11={fit_acc[1,1]:.4f}")
    print(f"  {'tau':>4} {'head0':>8} {'head1':>8}  (trained: {sorted(taus)})")
    curves = []
    for gi, tau in enumerate(grid):
        row = []
        for head in range(2):
            v = band_acc[gi][head]
            row.append(float(np.mean(v)) if v else float("nan"))
        curves.append(row)
        e0 = f"{row[0]:8.4f}" if row[0] == row[0] else "     n/a"
        e1 = f"{row[1]:8.4f}" if row[1] == row[1] else "     n/a"
        print(f"  {int(tau):>4} {e0} {e1}{' *' if int(tau) in taus else ''}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"ckpt": args.ckpt, "fit": fit_acc.tolist(), "diag": diag,
                   "offdiag": off, "delta": off - diag, "grid": grid,
                   "curves": curves}, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

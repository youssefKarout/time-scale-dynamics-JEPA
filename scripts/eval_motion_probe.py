"""Label-free functional downstream probe (no GT needed).

Frozen h0 -> ridge regression -> video motion energy, defined purely from RGB:
    y_tau = mean | X_{t+tau} - X_t |  (per clip, normalized pixels)

If h0 is a genuine dynamical state, it should predict near-future motion (tau=4)
and far-future motion (tau=16) above chance. Compares all 5 variants.
Usage:
    python scripts/eval_motion_probe.py --n-clips 128
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.movi_a_dataset import MoviADataset
from src.metrics.specialization import preprocess_pairs
from src.models.lts_dynamics_jepa import LatentTimeScaleJEPA
from src.utils.config import to_dotdict

CKPTS = {
    "primary": "runs/movi_a_poc/primary/ckpt_epoch99.pt",
    "shared": "runs/movi_a_poc/shared/ckpt_epoch99.pt",
    "shared_wide": "runs/movi_a_poc/shared_wide/ckpt_epoch99.pt",
    "horizon_cond": "runs/movi_a_poc/horizon_cond/ckpt_epoch99.pt",
    "regressor": "runs/movi_a_poc/regressor/ckpt_epoch99.pt",
}

def ridge_r2(Xtr, Ytr, Xte, Yte, alpha=1.0):
    Xb = np.concatenate([Xtr, np.ones((len(Xtr), 1))], 1)
    Xt = np.concatenate([Xte, np.ones((len(Xte), 1))], 1)
    A = Xb.T @ Xb + alpha * np.eye(Xb.shape[1])
    W = np.linalg.solve(A, Xb.T @ Ytr)
    P = Xt @ W
    ss = ((Yte - P) ** 2).sum()
    tot = ((Yte - Yte.mean(0, keepdims=True)) ** 2).sum()
    return float(1 - ss / (tot + 1e-8))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clips", type=int, default=128)
    ap.add_argument("--out", default="runs/movi_a_poc/motion_probe.json")
    a = ap.parse_args()
    # dataset for clip access (config root from primary ckpt)
    ck0 = torch.load(CKPTS["primary"], map_location="cpu", weights_only=False)
    cfg0 = to_dotdict(ck0["cfg"])
    ds = MoviADataset(root=cfg0.data.root, horizons=[4, 16],
                      num_clips=cfg0.data.num_clips, max_start_jitter=0,
                      train=False, val_fraction=cfg0.data.val_fraction)
    n = min(a.n_clips, len(ds))
    size = int(cfg0.data.frame_size)
    # motion targets from raw video (t = mid start like eval)
    Y = {4: [], 16: []}
    items = [ds[i] for i in range(n)]
    for b in items:
        clip = ds.load_clip(int(b["index"])).astype(np.float32) / 255.0
        t = int(b["t"])
        for tau in (4, 16):
            d = np.abs(clip[t + tau] - clip[t]).mean()
            Y[tau].append(d)
    Y = {k: np.array(v)[:, None] for k, v in Y.items()}
    print(f"motion-energy std: tau4={Y[4].std():.4f} tau16={Y[16].std():.4f} (need >0)")
    res = {}
    for name, ck in CKPTS.items():
        c = torch.load(ck, map_location="cpu", weights_only=False)
        cfg = to_dotdict(c["cfg"])
        m = LatentTimeScaleJEPA(cfg); m.load_state_dict(c["model"]); m.eval()
        with torch.no_grad():
            on = torch.stack([b["online"] for b in items])
            h0 = m.encode_online(preprocess_pairs(on, size)).numpy()
        # train/test split 80/20
        s = n * 4 // 5
        r = {}
        for tau in (4, 16):
            r[tau] = ridge_r2(h0[:s], Y[tau][:s], h0[s:], Y[tau][s:])
        res[name] = r
        print(f"  {name:12s} R2(tau4)={r[4]:+.3f}  R2(tau16)={r[16]:+.3f}  h0_std={h0.std():.4f}")
    import json
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")

if __name__ == "__main__":
    raise SystemExit(main())

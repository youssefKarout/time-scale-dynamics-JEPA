"""Rollout video: GT vs each variant, fixed anchor, per-object predicted centres.

WHAT IT SHOWS
-------------
From ONE anchor pair (X_{t-1}, X_t) at t = `--anchor`, roll each model's latent
forward to tau = 1..20 and decode *per-object* centre positions at every tau.

The decode uses a SINGLE ridge probe fit at "now" (h0 -> per-object centre at t),
then applied *unchanged* to every rolled latent.  This is deliberate: fitting a
separate probe per tau is tautological (each tau gets a free linear map, so every
model looks equally good), and fitting the probe on rolled latents lets the probe
absorb the dynamics.  One probe, calibrated at "now", measures the rollout.

Panels (2x2): GT | primary (ours) | shared_wide (B1-wide) | horizon_cond (B2).

  green  ring   = GT object centre at the displayed frame
  grey   trail  = GT trajectory of that object since the anchor
  coloured ring = model-predicted centre at t+tau (label = object index)

Reading a healthy model: predicted rings sit near the green GT rings and drift
smoothly with tau.  In our PoC, `primary` and `shared_wide` trace plausible short
arcs, whereas `horizon_cond` freezes (its encoder collapses across time).

NOTE (bug fixed): an earlier version fit the probe on the anchor latents and then
applied it to rolled latents, and used the *scene centre of mass* as the target
(nearly static).  Both made the prediction look frozen.  Do not reintroduce.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics.specialization import preprocess_pairs  # noqa: E402
from src.models.lts_dynamics_jepa import LatentTimeScaleJEPA  # noqa: E402
from src.utils.config import to_dotdict  # noqa: E402

KMAX = 8
COLS = [(255, 60, 60), (60, 255, 60), (90, 150, 255), (255, 190, 40),
        (255, 80, 255), (60, 230, 230), (190, 255, 190), (255, 160, 160)]
CK = {"primary": "runs/movi_a_poc/primary/ckpt_epoch99.pt",
      "shared_wide": "runs/movi_a_poc/shared_wide/ckpt_epoch99.pt",
      "horizon_cond": "runs/movi_a_poc/horizon_cond/ckpt_epoch99.pt"}


def load_rgb(rgb_dir, i):
    return np.load(os.path.join(rgb_dir, f"{i:05d}.npz"), allow_pickle=True)["video"]


def load_bb(gt_dir, i):
    return np.load(os.path.join(gt_dir, f"{i:05d}.npz"), allow_pickle=True)["bboxes"]


def centers_all(bb, T=24, kmax=KMAX):
    A = np.full((T, kmax, 2), np.nan)
    for k in range(min(len(bb), kmax)):
        a = np.asarray(bb[k])
        for F in range(min(T, a.shape[0])):
            y0, x0, y1, x1 = [float(v) for v in a[F]]
            if (x1 - x0) <= 0.005 or (y1 - y0) <= 0.005:
                continue
            A[F, k] = [(x0 + x1) / 2, (y0 + y1) / 2]
    return A


def roll_tau(m, H0, t):
    """Roll `m` from H0 to horizon t (field-1 for primary, its single field for shared)."""
    v = m.cfg.run.variant
    if v in ("primary", "shared", "shared_wide"):
        return m.integrate_field(1 if v == "primary" else 0, H0, float(t))
    return m.predict(H0, 1, float(t))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rgb-dir", default=None)
    ap.add_argument("--gt-dir", default=None)
    ap.add_argument("--out", default="runs/movi_a_poc/rollout_vs_others.gif")
    ap.add_argument("--n-fit-clips", type=int, default=60, help="clips used to fit the probe")
    ap.add_argument("--fit-anchors", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7])
    ap.add_argument("--clip", type=int, default=0, help="clip rendered")
    ap.add_argument("--anchor", type=int, default=2, help="anchor frame rendered")
    ap.add_argument("--taus", type=int, nargs="+", default=list(range(1, 21)))
    ap.add_argument("--alpha", type=float, default=1e-1)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--duration", type=int, default=550)
    args = ap.parse_args()

    c0 = torch.load(ROOT / CK["primary"], map_location="cpu", weights_only=False)
    cfg0 = to_dotdict(c0["cfg"])
    size = int(cfg0.data.frame_size)
    rgb_dir = args.rgb_dir or cfg0.data.root
    gt_dir = args.gt_dir or str(Path(rgb_dir).parent / "gt")

    M = {}
    for n, p in CK.items():
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = LatentTimeScaleJEPA(to_dotdict(ck["cfg"]))
        m.load_state_dict(ck["model"])
        m.eval()
        M[n] = m
    print(f"loaded {list(M)}  size={size}\nrgb={rgb_dir}\ngt ={gt_dir}", flush=True)

    clips = [load_rgb(rgb_dir, i) for i in range(args.n_fit_clips)]
    CT = [centers_all(load_bb(gt_dir, i)) for i in range(args.n_fit_clips)]

    pairs, Pa = [], []
    for ci in range(args.n_fit_clips):
        for a in args.fit_anchors:
            pair = np.stack([clips[ci][a - 1], clips[ci][a]], 0)
            pairs.append(torch.from_numpy(pair).permute(0, 3, 1, 2).float().unsqueeze(0) / 255.0)
            Pa.append((ci, a))

    ROLL, H0n = {}, {}
    with torch.no_grad():
        X = preprocess_pairs(torch.cat(pairs, 0), size)
        for n in M:
            h0 = M[n].encode_online(X)
            H0n[n] = h0.numpy()
            ROLL[n] = torch.stack([roll_tau(M[n], h0, t) for t in args.taus], 1).numpy()
            print(f"  {n:12s} rolled {ROLL[n].shape}", flush=True)

    Y0 = np.stack([CT[ci][a] for (ci, a) in Pa])
    Yt = {t: np.stack([CT[ci][a + t] if a + t < 24 else np.full((KMAX, 2), np.nan)
                       for (ci, a) in Pa]) for t in args.taus}

    W = {}
    for n in M:
        Xb = np.concatenate([H0n[n], np.ones((len(Pa), 1))], 1)
        Ws = []
        for k in range(KMAX):
            v = ~np.isnan(Y0[:, k, 0])
            Xv = Xb[v]
            Ws.append(np.linalg.solve(Xv.T @ Xv + args.alpha * np.eye(Xv.shape[1]),
                                      Xv.T @ Y0[v, k]) if v.sum() >= 10
                      else np.zeros((Xb.shape[1], 2)))
        W[n] = np.stack(Ws)
        pr = np.stack([Xb @ W[n][k] for k in range(KMAX)], 1)
        print(f"  {n:12s} probe-fit@now = "
              f"{np.nanmean(np.linalg.norm(pr - Y0, axis=2)) * 128:5.1f}px", flush=True)

    pidx = [i for i, (ci, a) in enumerate(Pa) if ci == args.clip and a == args.anchor][0]
    print(f"\nrollout error (px), clip {args.clip}, anchor t={args.anchor}", flush=True)
    for n in M:
        row = []
        for ti, t in enumerate(args.taus):
            Xb = np.concatenate([ROLL[n][pidx, ti, :], np.ones(1)])
            pr = np.stack([Xb @ W[n][k] for k in range(KMAX)])
            row.append(np.nanmean(np.linalg.norm(pr - Yt[t][pidx], axis=1)) * 128)
        print(f"  {n:12s} " + " ".join(f"{t}:{e:5.1f}" for t, e in zip(args.taus, row)), flush=True)

    clip = clips[args.clip]
    names = ["GT", "primary (ours)", "shared_wide", "horizon_cond"]
    SC = args.scale
    frames = []
    with torch.no_grad():
        for ti, t in enumerate(args.taus):
            F = args.anchor + t
            if F >= 24:
                break
            tiles = []
            for nm in names:
                img = Image.fromarray(clip[F]).resize((128 * SC, 128 * SC), Image.NEAREST)
                d = ImageDraw.Draw(img)
                gt = CT[args.clip][F]
                for k in range(KMAX):
                    if np.isnan(CT[args.clip][args.anchor][k, 0]):
                        continue
                    ax, ay = CT[args.clip][args.anchor][k] * 128 * SC
                    d.ellipse([ax - 3, ay - 3, ax + 3, ay + 3], outline=(150, 150, 150), width=1)
                for k in range(KMAX):
                    if np.isnan(gt[k, 0]):
                        continue
                    tr = [(CT[args.clip][tt][k, 0] * 128 * SC, CT[args.clip][tt][k, 1] * 128 * SC)
                          for tt in range(args.anchor, F + 1)
                          if not np.isnan(CT[args.clip][tt][k, 0])]
                    if len(tr) > 1:
                        d.line(tr, fill=(180, 180, 180), width=1)
                    gx, gy = gt[k] * 128 * SC
                    d.ellipse([gx - 6, gy - 6, gx + 6, gy + 6], outline=(0, 255, 0), width=2)
                if nm != "GT":
                    key = "primary" if nm.startswith("primary") else nm
                    Xb = np.concatenate([ROLL[key][pidx, ti, :], np.ones(1)])
                    for k in range(KMAX):
                        px, py = (Xb @ W[key][k]) * 128 * SC
                        c = COLS[k]
                        if 0 <= px < 128 * SC and 0 <= py < 128 * SC:
                            d.ellipse([px - 8, py - 8, px + 8, py + 8], outline=c, width=3)
                            d.text((px + 10, py - 6), f"{k}", fill=c)
                d.text((6, 6), f"{nm}  anchor t={args.anchor}  tau={t}  GT t={F}",
                       fill=(255, 255, 255))
                d.text((6, 22), "green=GT now  grey=GT trail  coloured=pred",
                       fill=(255, 255, 120))
                tiles.append(img)
            Wd, Ht = tiles[0].size
            cv = Image.new("RGB", (2 * Wd, 2 * Ht), (0, 0, 0))
            for j, tl in enumerate(tiles):
                cv.paste(tl, ((j % 2) * Wd, (j // 2) * Ht))
            frames.append(cv)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=args.duration, loop=0)
    print(f"\nsaved {out}  ({len(frames)} frames, {frames[0].size})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
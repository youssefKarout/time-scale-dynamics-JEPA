"""Diagnose whether the two per-horizon velocity fields are genuinely distinct.

Motivation
----------
The PoC claims (IDEA.md) that the two branches (tau1=4, tau2=16) develop
*different intrinsic temporal scales*.  The decisive failure mode is a GAUGE
DEGENERACY: because the ODE is integrated over NORMALISED time u in [0,1] and
the physical horizon enters only as an external multiplier (`dh/du = tau_k*V`),
the field network sees the SAME input (h, u) on both branches.  The objective is
then invariant to  V <- (1/a)V , tau <- a*tau , so gradient descent may settle on
a single dynamics law whose branches differ only by the fixed factor
tau2/tau1 = 4  ->  the method reduces *exactly* to the shared-field control B1.

This script tests that hypothesis directly, in latent space and in task space.

Evidence produced
-----------------
  (A) inter-field alignment: cos(V0(h,u), V1(h,u)) and ||V0||/||V1||
  (B) gauge test: does integrate(field_0, h0, tau/s) reproduce
      integrate(field_1, h0, tau)?  Report cos / rel-dist and the best s* per tau.
      s* == tau2/tau1 for every tau  =>  fields are one flow, rescaled.
  (C) rollout stability: relative displacement ||h(tau)-h0|| / ||h0|| per field.
  (D) task space (single ridge probe calibrated at "now"): per-tau ADE for
      field-0 vs field-1, i.e. the band test WITHOUT per-tau re-fitting.
  (E) trajectory of the degeneracy across training epochs.

Usage
-----
    python scripts/diagnose_field_degeneracy.py                     # all checks
    python scripts/diagnose_field_degeneracy.py --checks A B C
    python scripts/diagnose_field_degeneracy.py --epochs 0 10 99
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics.specialization import preprocess_pairs  # noqa: E402
from src.models.lts_dynamics_jepa import LatentTimeScaleJEPA  # noqa: E402
from src.utils.config import to_dotdict  # noqa: E402

KMAX = 8


def load_pairs(rgb_dir: str, n_clips: int, anchors, size: int) -> torch.Tensor:
    """Stack the (t-1, t) frame pairs for every (clip, anchor) into [P, 6, S, S]."""
    pairs = []
    for ci in range(n_clips):
        v = np.load(os.path.join(rgb_dir, f"{ci:05d}.npz"), allow_pickle=True)["video"]
        for a in anchors:
            pair = np.stack([v[a - 1], v[a]], 0)                 # [2, H, W, 3]
            pairs.append(torch.from_numpy(pair).permute(0, 3, 1, 2).float().unsqueeze(0) / 255.0)
    return preprocess_pairs(torch.cat(pairs, 0), size)           # [P, 6, S, S]


def load_rgb(rgb_dir: str, ci: int) -> np.ndarray:
    return np.load(os.path.join(rgb_dir, f"{ci:05d}.npz"), allow_pickle=True)["video"]


def centers_all(bb, T: int = 24, kmax: int = KMAX) -> np.ndarray:
    A = np.full((T, kmax, 2), np.nan)
    for k in range(min(len(bb), kmax)):
        a = np.asarray(bb[k])
        for F in range(min(T, a.shape[0])):
            y0, x0, y1, x1 = [float(v) for v in a[F]]
            if (x1 - x0) <= 0.005 or (y1 - y0) <= 0.005:
                continue
            A[F, k] = [(x0 + x1) / 2, (y0 + y1) / 2]
    return A


def build(rgb_dir: str, ckpt: str, n_clips: int, anchors, device: str = "cpu"):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = to_dotdict(ck["cfg"])
    size = int(cfg.data.frame_size)
    model = LatentTimeScaleJEPA(cfg)
    model.load_state_dict(ck["model"])
    model.eval().to(device)
    X = load_pairs(rgb_dir, n_clips, anchors, size).to(device)
    return model, cfg, size, X


def check_A(model, X, taus):
    """Inter-field alignment of the raw velocity at u=0 and u=1."""
    with torch.no_grad():
        h0 = model.encode_online(X)
        f0, f1 = model.fields[0], model.fields[1]
        align, norm, ratio = {}, {}, {}
        for u in (0.0, 1.0):
            uu = torch.full((h0.shape[0], 1), u)
            v0, v1 = f0(h0, uu), f1(h0, uu)
            align[u] = float(torch.nn.functional.cosine_similarity(v0, v1, dim=1).mean())
            n0, n1 = float(v0.norm(dim=1).mean()), float(v1.norm(dim=1).mean())
            norm[u] = (n0, n1); ratio[u] = n0 / n1
    print("\n(A) inter-field alignment at the anchor")
    print(f"    |h0| = {h0.norm(dim=1).mean():.3f}   (h0 per-dim std = {h0.std(0).mean():.4f})")
    for u in (0.0, 1.0):
        print(f"    u={u}: cos(v0,v1)={align[u]:+.4f}  ||v0||={norm[u][0]:.4f}  "
              f"||v1||={norm[u][1]:.4f}  ratio={ratio[u]:.3f}")
    print(f"    tau2/tau1 = {taus[1] / taus[0]:.3f}  <- if ratio ~ this AND cos ~ 1,")
    print("    the two fields are ONE flow with the fixed horizon scaling.")
    return align


def check_B(model, X, taus, scales):
    """Gauge test: f0 integrated to tau/s vs f1 integrated to tau."""
    with torch.no_grad():
        h0 = model.encode_online(X)
        R1 = {t: model.integrate_field(1, h0, float(t)) for t in taus}
        table = {}
        for t in taus:
            table[t] = {}
            for s in scales:
                R0 = model.integrate_field(0, h0, float(t) / s)
                table[t][s] = (float(torch.nn.functional.cosine_similarity(R0, R1[t], dim=1).mean()),
                               float(((R0 - R1[t]).norm(dim=1) / (R1[t].norm(dim=1) + 1e-6)).mean()))
    print("\n(B) gauge test:  cos / rel-dist  of  f0@(tau/s)  vs  f1@tau")
    print(f"    {'tau':>4} " + " ".join(f"{'s=' + str(s):>14}" for s in scales))
    for t in taus:
        print(f"    {t:>4} " + " ".join(f"{table[t][s][0]:+.3f}/{table[t][s][1]:5.2f}".rjust(14)
                                       for s in scales))
    print("    best scale s* matching f1@tau (max cosine):")
    for t in taus:
        s_best = max(scales, key=lambda s: table[t][s][0])
        print(f"      tau={t:>2}  s*={s_best:<6} cos={table[t][s_best][0]:+.4f}  "
              f"(tau_eff={t / s_best:.3f})")


def check_C(model, X, taus):
    """Rollout stability: how far each field travels per unit tau."""
    with torch.no_grad():
        h0 = model.encode_online(X)
        n0 = h0.norm(dim=1) + 1e-6
        out = {}
        for name, idx in (("f0", 0), ("f1", 1)):
            out[name] = [float(((model.integrate_field(idx, h0, float(t)) - h0).norm(dim=1) / n0).mean())
                         for t in taus]
    print("\n(C) rollout stability  ||h(tau)-h0|| / ||h0||   (growth = unstable field)")
    for name in ("f0", "f1"):
        print(f"    {name}: " + " ".join(f"{t}:{v:6.3f}" for t, v in zip(taus, out[name])))
    return out


def check_D(model, X, taus, gt_dir, n_clips, anchors, alpha):
    """Task-space band test: ONE probe calibrated at 'now', applied to both rollouts."""
    Y0, Yt = [], {t: [] for t in taus}
    for ci in range(n_clips):
        bb = np.load(os.path.join(gt_dir, f"{ci:05d}.npz"), allow_pickle=True)["bboxes"]
        C = centers_all(bb)
        for a in anchors:
            Y0.append(C[a])
            for t in taus:
                Yt[t].append(C[a + t] if a + t < 24 else np.full((KMAX, 2), np.nan))
    Y0 = np.stack(Y0)
    Yt = {t: np.stack(v) for t, v in Yt.items()}
    with torch.no_grad():
        h0 = model.encode_online(X)
        roll = {f: {t: model.integrate_field(0 if f == "f0" else 1, h0, float(t)) for t in taus}
                for f in ("f0", "f1")}
    Xb = np.concatenate([h0.cpu().numpy(), np.ones((len(h0), 1))], 1)
    W = []
    for k in range(KMAX):
        v = ~np.isnan(Y0[:, k, 0])
        Xv = Xb[v]
        W.append(np.linalg.solve(Xv.T @ Xv + alpha * np.eye(Xv.shape[1]), Xv.T @ Y0[v, k]))
    W = np.stack(W)
    pr = np.stack([Xb @ W[k] for k in range(KMAX)], 1)
    print("\n(D) task-space band test, probe fit ONCE at 'now' "
          f"(fit err {np.nanmean(np.linalg.norm(pr - Y0, axis=2)) * 128:.1f}px)")
    print(f"    {'tau':>4} {'f0 (px)':>10} {'f1 (px)':>10} {'adv f0-f1':>10}")
    res = {}
    for f in ("f0", "f1"):
        res[f] = []
        for t in taus:
            Rb = np.concatenate([roll[f][t].cpu().numpy(), np.ones((len(h0), 1))], 1)
            p = np.stack([Rb @ W[k] for k in range(KMAX)], 1)
            res[f].append(float(np.nanmean(np.linalg.norm(p - Yt[t], axis=2))) * 128)
    for i, t in enumerate(taus):
        print(f"    {t:>4} {res['f0'][i]:>10.1f} {res['f1'][i]:>10.1f} {res['f0'][i] - res['f1'][i]:>+10.1f}")
    return res


def check_E(rgb_dir, run_dir, epochs, n_clips, anchors, tau_hi):
    """Trajectory of the degeneracy across training epochs.

    Note: epochs 0-1 of `runs/movi_a_poc/primary` were overwritten by an
    accidental `--smoke` run (see FINDINGS §8), so they are excluded by default.
    """
    print(f"\n(E) degeneracy across training (gauge test at tau={tau_hi})")
    print(f"    {'ep':>4} {'cos(v0,v1)':>11} {'|v0|/|v1|':>10} {'cos(f0@t/4,f1@t)':>17} {'cos(f0@t,f1@t)':>15}")
    for ep in epochs:
        p = Path(run_dir) / f"ckpt_epoch{ep}.pt"
        if not p.exists():
            print(f"    {ep:>4}  (missing {p.name})")
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        cfg = to_dotdict(ck["cfg"])
        m = LatentTimeScaleJEPA(cfg)
        m.load_state_dict(ck["model"])
        m.eval()
        with torch.no_grad():
            X = load_pairs(rgb_dir, n_clips, anchors, int(cfg.data.frame_size))
            h0 = m.encode_online(X)
            u0 = torch.zeros(h0.shape[0], 1)
            v0, v1 = m.fields[0](h0, u0), m.fields[1](h0, u0)
            cs = float(torch.nn.functional.cosine_similarity(v0, v1, dim=1).mean())
            rr = float(v0.norm(dim=1).mean() / v1.norm(dim=1).mean())
            R1 = m.integrate_field(1, h0, float(tau_hi))
            c4 = float(torch.nn.functional.cosine_similarity(
                m.integrate_field(0, h0, float(tau_hi) / 4.0), R1, dim=1).mean())
            c1 = float(torch.nn.functional.cosine_similarity(
                m.integrate_field(0, h0, float(tau_hi)), R1, dim=1).mean())
        print(f"    {ep:>4} {cs:>11.4f} {rr:>10.3f} {c4:>17.4f} {c1:>15.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="runs/movi_a_poc/primary/ckpt_epoch99.pt")
    ap.add_argument("--run-dir", default="runs/movi_a_poc/primary")
    ap.add_argument("--rgb-dir", default=None)
    ap.add_argument("--gt-dir", default=None)
    ap.add_argument("--n-clips", type=int, default=64)
    ap.add_argument("--anchors", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7])
    ap.add_argument("--taus", type=int, nargs="+", default=[1, 2, 4, 8, 16, 20])
    ap.add_argument("--scales", type=float, nargs="+", default=[0.25, 1.0, 4.0, 16.0])
    ap.add_argument("--epochs", type=int, nargs="+", default=[5, 10, 20, 40, 60, 80, 99])
    ap.add_argument("--alpha", type=float, default=1e-1)
    ap.add_argument("--checks", nargs="+", default=["A", "B", "C", "D", "E"])
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = to_dotdict(ck["cfg"])
    rgb_dir = args.rgb_dir or cfg.data.root
    gt_dir = args.gt_dir or str(Path(rgb_dir).parent / "gt")
    trained = list(cfg.data.horizons)
    print(f"ckpt        : {args.ckpt}")
    print(f"variant     : {cfg.run.variant}")
    print(f"trained taus: {trained}   ratio = {trained[1] / trained[0]:.2f}")
    print(f"rgb / gt    : {rgb_dir} / {gt_dir}")
    print(f"n_clips={args.n_clips} anchors={args.anchors} taus={args.taus}")

    model, cfg, size, X = build(rgb_dir, args.ckpt, args.n_clips, args.anchors)
    if "A" in args.checks:
        check_A(model, X, trained)
    if "B" in args.checks:
        check_B(model, X, args.taus, args.scales)
    if "C" in args.checks:
        check_C(model, X, args.taus)
    if "D" in args.checks:
        check_D(model, X, args.taus, gt_dir, args.n_clips, args.anchors, args.alpha)
    if "E" in args.checks:
        check_E(rgb_dir, args.run_dir, args.epochs, min(args.n_clips, 48),
                args.anchors, trained[1])
    print("\nDONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
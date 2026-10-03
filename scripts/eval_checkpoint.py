"""Evaluate a trained checkpoint with the full specialisation metric suite.

Example:
    python scripts/eval_checkpoint.py --ckpt runs/movi_a_poc/primary/ckpt_epoch99.pt \
        --n-clips 128

Reconstructs the model from the checkpoint's saved config, then prints the
band-curve advantage (the decisive specialisation signal) plus all scalars.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.movi_a_dataset import MoviADataset  # noqa: E402
from src.metrics.specialization import specialization_report, summarize_report  # noqa: E402
from src.models.lts_dynamics_jepa import LatentTimeScaleJEPA  # noqa: E402
from src.utils.config import to_dotdict  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-clips", type=int, default=128)
    ap.add_argument("--val-fraction", type=float, default=None)
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = to_dotdict(ckpt["cfg"])

    model = LatentTimeScaleJEPA(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if args.val_fraction is not None:
        cfg.data.val_fraction = args.val_fraction

    # Force the decisive band grid even for checkpoints trained before it existed.
    # T=24 caps tau<=20 (t>=1, need t+tau<=23).
    cfg.metrics.band_horizons = [1, 2, 3, 4, 6, 8, 12, 16, 20]

    ds = MoviADataset(
        root=cfg.data.root,
        horizons=list(cfg.data.horizons),
        num_clips=cfg.data.num_clips,
        max_start_jitter=0,
        train=False,
        val_fraction=cfg.data.val_fraction,
    )

    report = specialization_report(model, ds, "cpu", cfg, n_clips=args.n_clips)
    scalars = summarize_report(report)

    def _fmt(v):
        return f"{v:+.4f}" if v is not None else "n/a"

    def _print_band(band, grid, taus):
        errs = band["errors"]
        adv = band["adv_1_over_0"]
        trained = {int(t) for t in taus}
        print(f"  {'tau':>4} {'err_f0':>8} {'err_f1':>8} {'adv(f1>f0)':>11}  (trained: {sorted(trained)})")
        for tau, e0, e1, a in zip(grid, errs[0], errs[1], adv):
            mark = " *" if int(tau) in trained else ""
            print(f"  {int(tau):>4} {e0:>8.4f} {e1:>8.4f} {a:>+11.4f}{mark}")
        print(f"  adv_short_mean={_fmt(band.get('adv_short_mean'))}  "
              f"adv_long_mean={_fmt(band.get('adv_long_mean'))}  "
              f"separation={_fmt(band.get('separation'))}  "
              f"crossing={band.get('crossing')}  "
              f"field1_wins_frac={band.get('field1_wins_frac')}")

    print(f"=== {args.ckpt}  (variant={cfg.run.variant}, n_clips={report['n_clips']}) ===")
    grid = report["band_grid"]
    for key in ("chtg/delta", "chtgstd/delta", "chtg/diag", "chtg/offdiag",
                "band/separation", "band/adv_short_mean", "band/adv_long_mean",
                "bandz/separation", "bandz/adv_short_mean", "bandz/adv_long_mean",
                "compose/gap", "repr/h0_std"):
        if key in scalars:
            print(f"  {key:24s} {scalars[key]:+.4f}")

    if report.get("band", {}).get("adv_1_over_0"):
        print("\n  [RAW space]")
        _print_band(report["band"], grid, report["taus"])

    if report.get("band_z", {}).get("adv_1_over_0"):
        print("\n  [Z-SCORED / objective-consistent space]")
        _print_band(report["band_z"], grid, report["taus"])

    print(f"\n  crossing(raw)={report['band'].get('crossing')}  "
          f"crossing(z)={report['band_z'].get('crossing')}  "
          f"field1_wins_frac(z)={report['band_z'].get('field1_wins_frac')}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"ckpt": args.ckpt, "scalars": scalars,
                       "band": report.get("band"),
                       "band_z": report.get("band_z"),
                       "chtg": report.get("chtg"),
                       "chtg_std": report.get("chtg_std")}, fh, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
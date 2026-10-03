"""Training entry point for LTS-D-JEPA (PoC on MOVi-A).

Usage:
    python -m src.train --config configs/movi_a_poc.yaml
    python -m src.train --config configs/movi_a_poc.yaml --set run.variant=shared
    python -m src.train --config configs/movi_a_poc.yaml --smoke
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data.movi_a_dataset import MoviADataset, collate_fn
from .data.transforms import augment_sample
from .metrics.specialization import specialization_report, summarize_report
from .models.lts_dynamics_jepa import LatentTimeScaleJEPA
from .utils.config import load_config, save_config
from .utils.seed import resolve_device, set_seed


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides key=value")
    ap.add_argument("--smoke", action="store_true", help="tiny fast run for validation")
    return ap.parse_args()


def apply_overrides(overrides):
    out = {}
    for item in overrides:
        k, v = item.split("=", 1)
        try:
            v = eval(v)  # noqa: S307 - trusted local CLI values (ints/lists/bools)
        except Exception:
            pass
        out[k] = v
    return out


def prepare_batch(batch, frame_size, device):
    """Apply per-sample augmentation and move to device."""
    online_u8 = batch["online"]                       # [B,2,3,H,W] uint8
    targets_u8 = batch["targets"]                     # list of [B,2,3,H,W]
    b = online_u8.shape[0]
    n_tau = len(targets_u8)

    online = []
    targets = [[] for _ in range(n_tau)]
    for i in range(b):
        o, ts = augment_sample(
            online_u8[i], [targets_u8[k][i] for k in range(n_tau)], frame_size, frame_size
        )
        online.append(o)
        for k in range(n_tau):
            targets[k].append(ts[k])

    return {
        "online": torch.stack(online).to(device),
        "targets": [torch.stack(t).to(device) for t in targets],
    }


def build_loaders(cfg, smoke=False):
    num_clips = 20 if smoke else cfg.data.num_clips
    train_ds = MoviADataset(
        root=cfg.data.root,
        horizons=cfg.data.horizons,
        num_clips=num_clips,
        max_start_jitter=cfg.data.max_start_jitter,
        train=True,
        val_fraction=cfg.data.val_fraction,
    )
    val_ds = MoviADataset(
        root=cfg.data.root,
        horizons=cfg.data.horizons,
        num_clips=num_clips,
        max_start_jitter=0,
        train=False,
        val_fraction=cfg.data.val_fraction,
    )
    bs = 8 if smoke else cfg.optim.batch_size
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=cfg.data.num_workers,
        collate_fn=collate_fn, drop_last=True,
    )
    return train_ds, val_ds, train_loader


def build_optimizer(model, cfg):
    enc_params = [p for n, p in model.encoder.named_parameters()
                  if p.requires_grad and "backbone" not in n]
    bb_params = [p for n, p in model.encoder.named_parameters()
                 if p.requires_grad and "backbone" in n]
    other = [p for n, p in model.named_parameters()
             if p.requires_grad and not n.startswith("encoder.")]
    groups = [
        {"params": enc_params, "lr": cfg.optim.lr_encoder},
        {"params": other, "lr": cfg.optim.lr_field},
    ]
    if bb_params:  # ViT backbone, unfrozen: train it at its own (small) LR
        groups.append({"params": bb_params, "lr": float(getattr(cfg.optim, "lr_backbone", 1e-5))})
    return torch.optim.AdamW(groups, weight_decay=cfg.optim.weight_decay)


def make_scheduler(optimizer, cfg, steps_per_epoch):
    warmup = max(1, cfg.optim.warmup_epochs * steps_per_epoch)
    total = max(warmup + 1, cfg.optim.epochs * steps_per_epoch)

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + torch.cos(torch.tensor(prog * 3.14159265)).item())

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    args = parse_args()
    cfg = load_config(args.config, apply_overrides(args.set))

    if args.smoke:
        cfg.data.num_clips = 20
        cfg.optim.epochs = 3
        cfg.optim.warmup_epochs = 1
        cfg.metrics.num_metric_clips = 16
        cfg.data.num_workers = 0
        # IMPORTANT: a smoke run must never write into a real run directory --
        # `out_dir` does not distinguish smoke from full, so without this the
        # tiny 3-epoch run would overwrite ckpt_epoch*.pt and train_log.csv of
        # the real run.
        if not cfg.run.name.endswith("_smoke"):
            cfg.run.name = f"{cfg.run.name}_smoke"

    set_seed(cfg.run.seed)
    device = resolve_device(cfg.run.device)
    print(f"[cfg] variant={cfg.run.variant} device={device} horizons={list(cfg.data.horizons)}")

    train_ds, val_ds, train_loader = build_loaders(cfg, smoke=args.smoke)
    print(f"[data] train clips={len(train_ds)} val clips={len(val_ds)} T={train_ds.T} "
          f"start_range=[{train_ds.t_min},{train_ds.t_max}]")

    model = LatentTimeScaleJEPA(cfg).to(device)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable params={n_param:,}")

    optimizer = build_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg, steps_per_epoch=max(1, len(train_loader)))

    use_amp = bool(cfg.optim.amp) and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast = torch.amp.autocast("cuda", enabled=use_amp)

    out_dir = Path(cfg.run.output_dir) / cfg.run.name / cfg.run.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")
    log_path = out_dir / "train_log.csv"
    log_file = open(log_path, "w", newline="")
    logger = csv.writer(log_file)

    metrics_rows = []

    for epoch in range(cfg.optim.epochs):
        model.train()
        t0 = time.time()
        running = {}
        for step, batch in enumerate(train_loader):
            batch = prepare_batch(batch, cfg.data.frame_size, device)

            optimizer.zero_grad(set_to_none=True)
            with autocast:
                out = model(batch)
                loss_dict = model.compute_loss(out)
                loss = loss_dict["total"]

            scaler.scale(loss).backward()
            if cfg.optim.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            model.ema()
            scheduler.step()

            for key, val in loss_dict["logs"].items():
                running[key] = running.get(key, 0.0) + float(val)
            running["steps"] = running.get("steps", 0) + 1

            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch} step {step}: {loss.item()}")

            if args.smoke and step >= 3:
                break

        n_steps = max(1, running.get("steps", 1))
        avg = {k: v / n_steps for k, v in running.items() if k != "steps"}
        dt = time.time() - t0
        msg = " ".join(f"{k}={v:.4f}" for k, v in avg.items())
        print(f"[epoch {epoch}] {msg} ({dt:.1f}s)")

        if not metrics_rows:
            logger.writerow(["epoch", "split", "key", "value"])
        for k, v in avg.items():
            logger.writerow([epoch, "train", k, f"{v:.6f}"])
        log_file.flush()

        torch.save({"model": model.state_dict(), "cfg": dict(cfg)}, out_dir / f"ckpt_epoch{epoch}.pt")

        if (epoch % cfg.metrics.every_n_epochs == 0) or (epoch == cfg.optim.epochs - 1) or args.smoke:
            report = specialization_report(model, val_ds, device, cfg)
            scalars = summarize_report(report)
            for k, v in scalars.items():
                logger.writerow([epoch, "metric", k, f"{v:.6f}"])
            log_file.flush()
            print(f"[metrics {epoch}] " + " ".join(f"{k}={v:.4f}" for k, v in scalars.items()))
            metrics_rows.append({"epoch": epoch, **scalars})

        if args.smoke and epoch >= 1:
            break

    log_file.close()
    print(f"[done] logs -> {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

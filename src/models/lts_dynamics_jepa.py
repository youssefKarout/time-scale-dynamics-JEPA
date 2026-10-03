"""Latent Time-Scale Dynamics JEPA.

Assembles the shared pair-encoder, the EMA target encoder, the per-branch
velocity fields (or control predictors), the target standardiser and the loss.

Variants (same code path, used for the control battery):
    primary      : separate velocity field V_{phi_k} per horizon      (method)
    shared       : one velocity field, integrated to both horizons    (B1)
    shared_wide  : B1 capacity-matched to primary (one field, wider)   (B1-wide)
    horizon_cond : single predictor g(h0, tau) — multi-horizon JEPA   (B2)
    regressor    : independent MLP per horizon, no dynamics           (B3)
"""
from __future__ import annotations

import copy
from typing import Dict, List

import torch
import torch.nn as nn

from .ema import copy_params, ema_update
from .encoder import build_encoder
from .integrator import integrate
from .standardizer import TargetStandardizer
from .velocity_field import TimeEmbedding, VelocityField, TauFiLM
from ..losses import jepa_distance, vicreg_loss


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.SiLU(inplace=True),
        nn.Linear(hidden, hidden),
        nn.SiLU(inplace=True),
        nn.Linear(hidden, out_dim),
    )


class LatentTimeScaleJEPA(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.variant = cfg.run.variant
        self.taus: List[int] = list(cfg.data.horizons)
        self.n = len(self.taus)
        d = cfg.model.latent_dim

        self.encoder = build_encoder(cfg)
        self.target_encoder = copy.deepcopy(self.encoder)
        copy_params(self.encoder, self.target_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        self.steps = cfg.model.integrator_steps
        self.scheme = cfg.model.integrator

        if self.variant == "primary":
            film_on = bool(getattr(cfg.model, "field_film", False))
            tau_fourier = int(getattr(cfg.model, "field_tau_fourier", 4))
            self.fields = nn.ModuleList(
                [VelocityField(d, cfg.model.field_hidden, cfg.model.field_time_dim, cfg.model.field_out_rank,
                               tau_dim=int(getattr(cfg.model, "field_tau_dim", 0)),
                               tau_max=float(max(self.taus)),
                               film=film_on, tau_fourier=tau_fourier)
                 for _ in range(self.n)]
            )
        elif self.variant in ("shared", "shared_wide"):
            # shared     : one velocity field, integrated to both horizons (B1)
            # shared_wide: same, but ~capacity-matched to primary's TWO fields
            #              (configurable multiplier; default 2x hidden width)
            mult = float(getattr(cfg.model, "shared_wide_mult", 2.0)) if self.variant == "shared_wide" else 1.0
            hidden = max(8, int(round(cfg.model.field_hidden * mult)))
            self.fields = nn.ModuleList(
                [VelocityField(d, hidden, cfg.model.field_time_dim, cfg.model.field_out_rank,
                               tau_dim=int(getattr(cfg.model, "field_tau_dim", 0)),
                               tau_max=float(max(self.taus)),
                               film=bool(getattr(cfg.model, "field_film", False)),
                               tau_fourier=int(getattr(cfg.model, "field_tau_fourier", 4)))]
            )
        elif self.variant == "horizon_cond":
            self.tau_embed = TimeEmbedding(cfg.model.field_time_dim)
            self.predictor = _mlp(d + self.tau_embed.out_dim, cfg.model.field_hidden, d)
        elif self.variant == "regressor":
            self.predictors = nn.ModuleList([_mlp(d, cfg.model.field_hidden, d) for _ in range(self.n)])
        else:
            raise ValueError(f"unknown variant {self.variant!r}")

        self.standardizer = (
            TargetStandardizer(self.n, d) if cfg.model.standardize_targets else None
        )

    # ------------------------------------------------------------------ encode
    def encode_online(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    @torch.no_grad()
    def encode_target(self, x: torch.Tensor) -> torch.Tensor:
        return self.target_encoder(x)

    # ----------------------------------------------------------------- predict
    def field(self, k: int) -> VelocityField:
        """Return the velocity field responsible for branch k (for metrics)."""
        if self.variant == "primary":
            return self.fields[k]
        if self.variant in ("shared", "shared_wide"):
            return self.fields[0]
        raise AttributeError(f"variant {self.variant!r} has no velocity field")

    def standardize(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """Apply branch-k target statistics (objective-consistent space)."""
        if self.standardizer is None:
            return x
        mu = self.standardizer.mean[k].detach()
        sc = self.standardizer.scale[k].detach().clamp_min(self.standardizer.eps)
        return (x - mu) / sc

    def predict(self, h0: torch.Tensor, k: int, tau: float | None = None) -> torch.Tensor:
        tau = float(self.taus[k] if tau is None else tau)
        degauged = bool(getattr(self.cfg.model, "field_tau_dim", 0))
        if self.variant in ("primary", "shared", "shared_wide"):
            return integrate(self.field(k), h0, tau, steps=self.steps, scheme=self.scheme,
                             tau_cond=(tau if degauged else None))
        if self.variant == "horizon_cond":
            u = torch.full((h0.shape[0], 1), tau / max(self.taus), dtype=h0.dtype, device=h0.device)
            t = self.tau_embed(u)
            return self.predictor(torch.cat([h0, t], dim=-1))
        if self.variant == "regressor":
            return self.predictors[k](h0)
        raise ValueError(self.variant)

    def integrate_field(self, field_idx: int, h0: torch.Tensor, tau: float, tau_cond: float | None = None):
        """Integrate field `field_idx` to an arbitrary horizon tau (metrics)."""
        if self.variant == "primary":
            fld = self.fields[field_idx]
        elif self.variant in ("shared", "shared_wide"):
            fld = self.fields[0]
        else:
            raise AttributeError(f"variant {self.variant!r} has no velocity field")
        if tau_cond is None and bool(getattr(self.cfg.model, "field_tau_dim", 0)):
            tau_cond = self.taus[field_idx] if self.variant == "primary" else float(tau)
        return integrate(fld, h0, float(tau), steps=self.steps, scheme=self.scheme, tau_cond=tau_cond)

    # ------------------------------------------------------------------ forward
    def forward(self, batch: Dict[str, object]) -> Dict[str, object]:
        from ..data.transforms import stack_pair_to_channels

        online = stack_pair_to_channels(batch["online"])          # [B,6,H,W]
        h0 = self.encode_online(online)

        preds, targets = [], []
        for k in range(self.n):
            preds.append(self.predict(h0, k))
            tgt = stack_pair_to_channels(batch["targets"][k])
            targets.append(self.encode_target(tgt))

        if self.standardizer is not None:
            self.standardizer.update(targets)
            preds_s, targets_s = self.standardizer(preds, targets)
        else:
            preds_s, targets_s = preds, targets

        return {"h0": h0, "preds": preds_s, "targets": targets_s,
                "preds_raw": preds, "targets_raw": targets}

    # --------------------------------------------------------------------- loss
    def compute_loss(self, out: Dict[str, object]) -> Dict[str, torch.Tensor]:
        lc = self.cfg.loss
        weights = lc.branch_weights or [1.0] * self.n
        total = torch.zeros((), device=out["h0"].device)
        logs: Dict[str, torch.Tensor] = {}
        for k in range(self.n):
            dk = jepa_distance(out["preds"][k], out["targets"][k], kind=lc.distance)
            total = total + float(weights[k]) * dk
            logs[f"loss_branch{k}"] = dk.detach()

        if (lc.vicreg_var > 0 or lc.vicreg_cov > 0):
            vreg = torch.zeros((), device=out["h0"].device)
            for k in range(self.n):
                vreg = vreg + vicreg_loss(
                    out["preds"][k], lc.vicreg_var, lc.vicreg_cov, lc.vicreg_gamma, lc.vicreg_eps
                )[0]
            total = total + vreg / self.n
            logs["loss_vicreg"] = (vreg / self.n).detach()

        logs["loss_total"] = total.detach()
        return {"total": total, "logs": logs}

    # ---------------------------------------------------------------------- ema
    @torch.no_grad()
    def ema(self) -> None:
        ema_update(self.encoder, self.target_encoder, self.cfg.model.ema_beta)

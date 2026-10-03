"""Learned latent velocity fields V_{phi_k}(h, u).

The field is defined on NORMALISED integration time u in [0, 1]; the physical
horizon enters through the integrator scaling (dh/du = tau * V(h, u)). This
keeps the time-embedding range fixed across branches while the horizon is
still explicit (V is reparametrised, still a function of physical tau).

NOTE: with tau_dim=0 both branches see identical inputs (h, u) and collapse to
a single rescaled flow (see FINDINGS_movi_a_no_specialization.md). Set tau_dim>0
(concat) or field_film=true (FiLM) to condition on the physical scale.

The last projection is initialised near zero so the flow starts close to the
identity, which keeps the backprop-through-solver path stable early in training.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    """Fixed sinusoidal embedding of normalised time u in [0, 1]."""

    def __init__(self, n_freq: int = 16) -> None:
        super().__init__()
        freqs = torch.exp(torch.linspace(0.0, math.log(50.0), n_freq))
        self.register_buffer("freqs", freqs)  # [n_freq]

    @property
    def out_dim(self) -> int:
        return self.freqs.numel() * 2

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: scalar, [B] or [B,1] -> [B, 2*n_freq]
        if u.dim() == 0:
            u = u.view(1, 1)
        elif u.dim() == 1:
            u = u.view(-1, 1)
        ang = 2.0 * math.pi * u * self.freqs.view(1, -1)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class TauFiLM(nn.Module):
    """Scale conditioning: (gamma(tau), beta(tau)) modulating hidden units.

    log(tau) -> Fourier features -> zero-init linear (identity at start).
    """

    def __init__(self, hidden: int, n_fourier: int = 4, tau_max: float = 16.0) -> None:
        super().__init__()
        self.n_fourier = max(1, int(n_fourier))
        self.tau_norm = float(__import__("math").log(max(tau_max, 2.0)))
        freqs = torch.exp(torch.linspace(0.0, math.log(8.0), self.n_fourier))
        self.register_buffer("freqs", freqs)  # [F]
        self.gen = nn.Linear(2 * self.n_fourier, 2 * hidden)
        nn.init.zeros_(self.gen.weight)
        nn.init.zeros_(self.gen.bias)

    @property
    def out_dim(self) -> int:
        return 2 * self.n_fourier

    def embed(self, tau: torch.Tensor | float | None, ref: torch.Tensor) -> torch.Tensor:
        """Fourier embedding [B, 2F] of normalised log(tau); zeros when None."""
        if tau is None:
            return ref.new_zeros(ref.shape[0], 2 * self.n_fourier)
        if not torch.is_tensor(tau):
            tau = torch.full((ref.shape[0], 1), float(tau), dtype=ref.dtype, device=ref.device)
        tau = tau.reshape(-1, 1).to(dtype=ref.dtype, device=ref.device)
        lt = torch.log(tau.clamp_min(1e-3)) / self.tau_norm          # [B,1] ~ [0,1]
        ang = 2.0 * math.pi * lt * self.freqs.view(1, -1)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def forward(self, tau: torch.Tensor | float | None, ref: torch.Tensor):
        gb = self.gen(self.embed(tau, ref))                          # [B, 2H]
        gamma, beta = gb.chunk(2, dim=-1)
        return gamma, beta


class VelocityField(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        hidden: int = 512,
        time_dim: int = 16,
        out_rank: int = 0,
        tau_dim: int = 0,       # >0 => de-gauged: extra log(tau) input channel
        tau_max: float = 16.0,  # normaliser for log(tau)
        film: bool = False,     # True => FiLM modulation from tau (strong conditioning)
        tau_fourier: int = 4,   # #Fourier freqs for the FiLM tau embedding
    ) -> None:
        super().__init__()
        self.time = TimeEmbedding(time_dim)
        self.tau_dim = int(tau_dim)
        self.tau_norm = float(__import__("math").log(max(tau_max, 2.0)))
        self.film_on = bool(film) and self.tau_dim > 0
        self.film = TauFiLM(hidden, n_fourier=int(tau_fourier), tau_max=tau_max) if self.film_on else None
        in_dim = latent_dim + self.time.out_dim + (self.tau_dim if not self.film_on else 0)
        self.net1 = nn.Linear(in_dim, hidden)
        self.act1 = nn.SiLU(inplace=True)
        self.net2 = nn.Linear(hidden, hidden)
        self.act2 = nn.SiLU(inplace=True)
        self.out_rank = int(out_rank)
        if self.out_rank > 0:  # capacity ladder (mechanism M2)
            self.head = nn.Sequential(nn.Linear(hidden, self.out_rank), nn.Linear(self.out_rank, latent_dim))
        else:
            self.head = nn.Linear(hidden, latent_dim)

        self._init_near_identity()

    def _init_near_identity(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.05)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # scale the final projection towards zero
        last = self.head[-1] if isinstance(self.head, nn.Sequential) else self.head
        with torch.no_grad():
            last.weight.mul_(0.05)
            if last.bias is not None:
                last.bias.zero_()

    def forward(self, h: torch.Tensor, u: torch.Tensor, tau: torch.Tensor | float | None = None) -> torch.Tensor:
        """h: [B, d]; u: scalar | [B] | [B,1]; tau: optional physical scale.

        De-gauging: when ``tau`` is given, the field conditions on log(tau)
        (normalised by log(tau_max)) so branches become distinguishable.
        When None (legacy), behaviour is exactly the old gauge-degenerate one.
        """
        t = self.time(u)
        if t.shape[0] != h.shape[0]:
            t = t.expand(h.shape[0], -1)
        feats = [h, t]
        if getattr(self, "tau_dim", 0) > 0 and self.film is None:
            # Concat path only; FiLM mode carries tau multiplicatively instead.
            if tau is None:
                # No scale info: zero-pad to match the linear layer.
                feats.append(h.new_zeros(h.shape[0], self.tau_dim))
            else:
                if not torch.is_tensor(tau):
                    tau = torch.full((h.shape[0], 1), float(tau), dtype=h.dtype, device=h.device)
                tau = tau.reshape(-1, 1).to(dtype=h.dtype, device=h.device)
                lt = torch.log(tau.clamp_min(1e-3)) / self.tau_norm
                feats.append(lt.expand(h.shape[0], -1))
        z = self.act1(self.net1(torch.cat(feats, dim=-1)))
        if self.film is not None:
            gamma, beta = self.film(tau, h)
            z = z * (1.0 + gamma) + beta
        z = self.act2(self.net2(z))
        return self.head(z)

    def tau_usage(self) -> dict:
        """Mean |weight| of the tau-conditioning path (diagnostic)."""
        out = {}
        with torch.no_grad():
            w1 = self.net1.weight  # [H, in]
            out["l1_norm"] = float(w1.abs().mean())
            if getattr(self, "tau_dim", 0) > 0 and not self.film_on:
                out["tau_col_norm"] = float(w1[:, -self.tau_dim:].abs().mean())
                out["trunk_col_norm"] = float(w1[:, :-self.tau_dim].abs().mean())
            if self.film is not None:
                out["film_gen_norm"] = float(self.film.gen.weight.abs().mean())
        return out

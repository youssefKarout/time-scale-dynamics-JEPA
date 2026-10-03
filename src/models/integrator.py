"""Fixed-step ODE integrators over normalised time u in [0, 1].

Physical dynamics:  dh/dtau = V(h, tau),  h_k(0) = h0.
Reparametrise tau = tau_k * u (u in [0,1]):  dh/du = tau_k * V(h, u).
Integrated with RK4/Euler; gradients flow through the solver.
"""
from __future__ import annotations

from typing import Callable, List, Tuple

import torch

FieldFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def integrate(
    field: FieldFn,
    h0: torch.Tensor,
    tau: float,
    steps: int = 8,
    scheme: str = "rk4",
    return_traj: bool = False,
    tau_cond: float | None = None,
) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """Integrate `field` from h0 over physical horizon `tau`.

    Returns h(tau), or (h(tau), traj, speeds) when return_traj is True, where
    traj[i] is the state at step i (len steps+1) and speeds[i] = ||V(h_i, u_i)||.

    `tau_cond`: when not None, forwarded as ``field(h, u, tau_cond)`` so a
    de-gauged field can condition on the physical scale; otherwise the legacy
    ``field(h, u)`` call is used (fully backward compatible).
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    du = 1.0 / steps
    h = h0
    traj: List[torch.Tensor] = [h]
    speeds: List[torch.Tensor] = []

    def _call(hh: torch.Tensor, uu: torch.Tensor) -> torch.Tensor:
        if tau_cond is None:
            return field(hh, uu)
        try:
            return field(hh, uu, tau_cond)
        except TypeError:
            return field(hh, uu)

    u = 0.0
    for _ in range(steps):
        u_t = torch.full((h.shape[0], 1), u, dtype=h.dtype, device=h.device)
        v0 = _call(h, u_t)
        speeds.append(v0.detach().norm(dim=-1))

        if scheme == "euler":
            h = h + du * (tau * v0)
        elif scheme == "rk4":
            u_mid = torch.full_like(u_t, u + 0.5 * du)
            u_end = torch.full_like(u_t, u + du)
            k1 = tau * v0
            k2 = tau * _call(h + 0.5 * du * k1, u_mid)
            k3 = tau * _call(h + 0.5 * du * k2, u_mid)
            k4 = tau * _call(h + du * k3, u_end)
            h = h + (du / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:
            raise ValueError(f"unknown scheme {scheme!r}")

        traj.append(h)
        u += du

    if return_traj:
        return h, traj, speeds
    return h


def path_length(speeds: List[torch.Tensor], tau: float, steps: int) -> torch.Tensor:
    """Approximate integral of ||V|| dtau = (tau/steps) * sum_i ||V_i||."""
    if not speeds:
        return torch.zeros(())
    stacked = torch.stack(speeds, dim=0)  # [steps, B]
    return stacked.sum(dim=0) * (tau / steps)

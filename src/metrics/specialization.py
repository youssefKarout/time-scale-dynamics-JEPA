"""Temporal-scale specialization metrics (label-free).

Def 1  speed & straightness       -> branch_speed_straightness
Def 2  cross-horizon transfer gap -> cross_horizon_gap   (CHTG, headline)
Def 3  temporal receptive band    -> band_curves
Def 4  flow systematicity         -> composition_test

specialization_report() assembles all of the above for the controls too; it
returns None for any metric a non-dynamical variant (B2/B3) cannot define.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..data.transforms import normalize, stack_pair_to_channels, to_float
from ..losses import jepa_distance


def preprocess_pairs(frames: torch.Tensor, size: int) -> torch.Tensor:
    """frames: [N, 2, 3, H, W] uint8 -> normalised [N, 6, S, S] (deterministic)."""
    x = to_float(frames)  # [N,2,3,H,W]
    if x.shape[-1] != size:
        n, f, c, h, w = x.shape
        x = F.interpolate(x.reshape(-1, c, h, w), size=(size, size), mode="bilinear", align_corners=False)
        x = x.reshape(n, f, c, size, size)
    x = normalize(x)
    return stack_pair_to_channels(x)


@torch.no_grad()
def branch_speed_straightness(model, h0: torch.Tensor) -> Dict[str, List[float]]:
    """Def 1. h0: [B, d]. Returns per-branch speed/straightness summaries.

    straightness = ||h(tau) - h(0)|| / (total path length of the latent
    trajectory). 1 => perfectly straight flow; <1 => curved/structured flow.
    """
    from ..models.integrator import integrate

    out: Dict[str, List[float]] = {"speed0": [], "path": [], "displacement": [], "straightness": []}
    degauged = bool(getattr(getattr(model, "cfg", None), "model", None) is not None
                    and getattr(model.cfg.model, "field_tau_dim", 0))
    for k in range(model.n):
        tau = float(model.taus[k])
        field = model.field(k)
        u0 = torch.zeros((h0.shape[0], 1), dtype=h0.dtype, device=h0.device)
        try:
            speed0 = field(h0, u0, tau if degauged else None).norm(dim=-1).mean().item()
        except TypeError:
            speed0 = field(h0, u0).norm(dim=-1).mean().item()
        h_end, traj, speeds = integrate(
            field, h0, tau, steps=model.steps, scheme=model.scheme, return_traj=True,
            tau_cond=(tau if degauged else None),
        )
        seg = torch.stack([(traj[i + 1] - traj[i]).norm(dim=-1) for i in range(len(traj) - 1)], dim=0)
        path = seg.sum(dim=0).mean().item()
        disp = (h_end - h0).norm(dim=-1).mean().item()
        out["speed0"].append(speed0)
        out["path"].append(path)
        out["displacement"].append(disp)
        out["straightness"].append(disp / (path + 1e-8))
    return out


@torch.no_grad()
def cross_horizon_gap(model, h0: torch.Tensor, target_feats: List[torch.Tensor],
                      standardize: bool = False) -> Dict[str, object]:
    """Def 2. target_feats[k] = E_xi(target pair at trained horizon tau_k).

    standardize=True evaluates in the model's objective-consistent space
    (branch-j statistics applied to both prediction and target), so raw-space
    geometry cannot create artefacts.
    """
    n = model.n
    e = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            pred = model.integrate_field(i, h0, float(model.taus[j]))
            tgt = target_feats[j]
            if standardize:
                pred = model.standardize(pred, j)
                tgt = model.standardize(tgt, j)
            e[i, j] = jepa_distance(pred, tgt, kind="cosine").item()
    diag = np.mean(np.diag(e))
    off = (e.sum() - np.trace(e)) / (n * (n - 1)) if n > 1 else diag
    return {"matrix": e.tolist(), "diag": float(diag), "offdiag": float(off), "delta": float(off - diag)}


@torch.no_grad()
def field_state_dependence(model, h0: torch.Tensor) -> Dict[str, object]:
    """Diagnostic: is V state-dependent, or a near-constant linear drift?

    Reports, per field, the across-batch spread of V(h0, 0) relative to its
    mean magnitude (``variation``; ~0 => constant/linear field), and the
    pairwise cosine alignment between the two fields' velocities.
    """
    if model.n < 1:
        return {}
    try:
        model.field(0)
    except AttributeError:
        return {}

    out: Dict[str, object] = {"variation": [], "norm": [], "tau_usage": []}
    vs = []
    for k in range(model.n):
        u0 = torch.zeros((h0.shape[0], 1), dtype=h0.dtype, device=h0.device)
        try:
            tau_c = float(model.taus[k]) if bool(getattr(getattr(model, "cfg", None), "model", None) is not None and getattr(model.cfg.model, "field_tau_dim", 0)) else None
        except (AttributeError, IndexError, TypeError):
            tau_c = None
        fld = model.field(k)
        try:
            v = fld(h0, u0, tau_c)
        except TypeError:
            v = fld(h0, u0)
        vs.append(v)
        norm = v.norm(dim=-1).mean().item()
        spread = v.std(dim=0).mean().item()
        out["variation"].append(spread / (norm + 1e-8))
        out["norm"].append(norm)
        try:
            out["tau_usage"].append(fld.tau_usage() if hasattr(fld, "tau_usage") else {})
        except Exception:
            out["tau_usage"].append({})

    if len(vs) >= 2:
        cos = F.cosine_similarity(vs[0], vs[1], dim=-1).mean().item()
        out["field_cos01"] = cos
        out["norm_ratio01"] = out["norm"][0] / (out["norm"][1] + 1e-8)
    return out


@torch.no_grad()
def band_curves(model, h0: torch.Tensor, grid: List[int], target_feats_by_tau: Dict[int, torch.Tensor],
                zscore: bool = False):
    """Def 3. target_feats_by_tau[tau] = E_xi(target pair at horizon tau).

    zscore=True evaluates in an objective-consistent space: each horizon's
    target set is centred by its own batch mean (stop-grad) and the same shift is
    applied to predictions before the (scale-invariant) cosine distance. This
    removes the per-horizon magnitude/centring confound that raw space carries.

    Returns per-field error curves over the (held-out) horizon grid.
    """
    n = model.n
    grid = list(grid)
    errs = np.zeros((n, len(grid)), dtype=np.float64)
    for i in range(n):
        for gi, tau in enumerate(grid):
            tgt = target_feats_by_tau[tau]
            pred = model.integrate_field(i, h0, float(tau))
            if zscore:
                mu = tgt.mean(dim=0, keepdim=True).detach()
                pred = pred - mu
                tgt = tgt - mu
            errs[i, gi] = jepa_distance(pred, tgt, kind="cosine").item()
    argmin = [grid[int(np.argmin(errs[i]))] for i in range(n)]

    # Field-1 advantage over field-0 at every (mostly held-out) horizon:
    #   adv(tau) = err_field0(tau) - err_field1(tau)
    # adv > 0  => the long-horizon field is BETTER at tau.
    # Genuine scale specialisation => adv(tau) < 0 for small tau AND > 0 for large
    # (a crossing). If adv keeps one sign across the held-out grid, one field
    # simply dominates and there is NO band.
    out: Dict[str, object] = {"grid": grid, "errors": errs.tolist(), "argmin_tau": argmin}
    if n >= 2:
        adv = (errs[0] - errs[1]).tolist()
        out["adv_1_over_0"] = adv
        trained = {int(t) for t in model.taus}
        held = [(gi, t) for gi, t in enumerate(grid) if t not in trained]
        t0 = float(model.taus[0])
        short = [adv[gi] for gi, t in held if t < t0]
        long_ = [adv[gi] for gi, t in held if t > t0]
        if short:
            out["adv_short_mean"] = float(np.mean(short))
        if long_:
            out["adv_long_mean"] = float(np.mean(long_))
        if short and long_:
            out["separation"] = float(np.mean(long_) - np.mean(short))

        held_adv = [adv[gi] for gi, _ in held]
        has_pos = any(a > 1e-6 for a in held_adv)
        has_neg = any(a < -1e-6 for a in held_adv)
        out["crossing"] = bool(has_pos and has_neg)          # True => band-like
        out["field1_wins_frac"] = float(np.mean([a > 0 for a in held_adv]))
        for tau_k in [int(t) for t in model.taus]:
            if tau_k in grid:
                out[f"adv_at_tau{tau_k}"] = float(adv[grid.index(tau_k)])
    return out


@torch.no_grad()
def composition_test(model, h0: torch.Tensor, target_feats: List[torch.Tensor]) -> Dict[str, float]:
    """Def 4. Re-apply the short field in tau_1-sized chunks up to tau_2.

    If branch 2 were simply the short field composed, its error would match;
    a large gap indicates non-nested dynamics (genuine scale separation).
    """
    if model.n < 2:
        return {}
    from ..models.integrator import integrate

    tau1, tau2 = float(model.taus[0]), float(model.taus[1])
    n_chunks = max(1, int(round(tau2 / tau1)))
    field0 = model.field(0)
    h = h0
    for _ in range(n_chunks):
        h = integrate(field0, h, tau1, steps=model.steps, scheme=model.scheme)
    composed_err = jepa_distance(h, target_feats[1], kind="cosine").item()

    direct = model.integrate_field(1, h0, tau2)
    direct_err = jepa_distance(direct, target_feats[1], kind="cosine").item()
    return {"composed_short_to_tau2": composed_err, "direct_long": direct_err,
            "gap": composed_err - direct_err}


def _pairs_to_model_input(pairs: np.ndarray) -> torch.Tensor:
    """pairs: [N,2,H,W,3] uint8 -> [N,2,3,H,W] uint8."""
    return torch.from_numpy(pairs).permute(0, 1, 4, 2, 3).contiguous()


@torch.no_grad()
def specialization_report(model, dataset, device, cfg, n_clips: Optional[int] = None, seed: int = 0) -> Dict[str, object]:
    """Compute the full Def 1-4 report (returns None for field-based metrics
    when the variant has no velocity field, e.g. controls B2/B3)."""
    model.eval()
    taus = list(model.taus)
    size = int(cfg.data.frame_size)
    n = min(int(n_clips or cfg.metrics.num_metric_clips), len(dataset))
    rng = np.random.RandomState(seed)
    idxs = rng.choice(len(dataset), size=n, replace=False)

    band_grid = cfg.metrics.band_horizons
    band_grid = list(range(1, max(taus) + 1)) if band_grid is None else [int(x) for x in band_grid]
    max_band = max(band_grid)
    t_max = dataset.T - 1 - max(max_band, max(taus))
    if t_max < 1:
        raise ValueError(f"clip length {dataset.T} too short for band grid up to {max_band}")

    online, tgt_by_tau = [], {tau: [] for tau in set(taus) | set(band_grid)}
    for idx in idxs:
        clip = dataset.load_clip(int(idx))
        t = int(rng.randint(1, t_max + 1))
        online.append(np.stack([clip[t - 1], clip[t]], 0))
        for tau in tgt_by_tau:
            tgt_by_tau[tau].append(np.stack([clip[t + tau - 1], clip[t + tau]], 0))

    online_t = _pairs_to_model_input(np.stack(online, 0)).to(device)
    h0 = model.encode_online(preprocess_pairs(online_t, size))

    feats = {}
    for tau, arrs in tgt_by_tau.items():
        x = _pairs_to_model_input(np.stack(arrs, 0)).to(device)
        feats[tau] = model.encode_target(preprocess_pairs(x, size))

    report: Dict[str, object] = {"n_clips": n, "taus": taus, "band_grid": band_grid}

    # collapse diagnostics: per-dim std of h0 and of each target feature
    report["h0_std"] = float(h0.std(dim=0).mean().item())
    report["target_std"] = [float(feats[tau].std(dim=0).mean().item()) for tau in taus]

    trained = [feats[tau] for tau in taus]
    try:
        report["speed"] = branch_speed_straightness(model, h0)
        report["chtg"] = cross_horizon_gap(model, h0, trained)
        report["chtg_std"] = cross_horizon_gap(model, h0, trained, standardize=True)
        report["fields"] = field_state_dependence(model, h0)
        report["band"] = band_curves(model, h0, band_grid, {tau: feats[tau] for tau in band_grid})
        report["band_z"] = band_curves(model, h0, band_grid, {tau: feats[tau] for tau in band_grid}, zscore=True)
        report["composition"] = composition_test(model, h0, trained)
    except AttributeError:
        report["speed"] = report["chtg"] = report["chtg_std"] = report["fields"] = None
        report["band"] = report["band_z"] = report["composition"] = None

    model.train()
    return report


def summarize_report(report: Dict[str, object]) -> Dict[str, float]:
    """Flatten a report into scalar logs for CSV/tensorboard."""
    out: Dict[str, float] = {}
    if report.get("h0_std") is not None:
        out["repr/h0_std"] = float(report["h0_std"])
        for i, s in enumerate(report.get("target_std", [])):
            out[f"repr/target_std{i}"] = float(s)
    if report.get("chtg"):
        out["chtg/delta"] = float(report["chtg"]["delta"])
        out["chtg/diag"] = float(report["chtg"]["diag"])
        out["chtg/offdiag"] = float(report["chtg"]["offdiag"])
    if report.get("chtg_std"):
        out["chtgstd/delta"] = float(report["chtg_std"]["delta"])
        out["chtgstd/diag"] = float(report["chtg_std"]["diag"])
        out["chtgstd/offdiag"] = float(report["chtg_std"]["offdiag"])
    if report.get("fields"):
        for i, v in enumerate(report["fields"].get("variation", [])):
            out[f"field/var{i}"] = float(v)
        if "field_cos01" in report["fields"]:
            out["field/cos01"] = float(report["fields"]["field_cos01"])
            out["field/norm_ratio01"] = float(report["fields"]["norm_ratio01"])
        for i, tu in enumerate(report["fields"].get("tau_usage", [])):
            if isinstance(tu, dict):
                for k2, v2 in tu.items():
                    try:
                        out[f"tauuse/f{i}_{k2}"] = float(v2)
                    except (TypeError, ValueError):
                        pass
    if report.get("band"):
        for i, am in enumerate(report["band"]["argmin_tau"]):
            out[f"band/argmin_field{i}"] = float(am)
        # full per-field error curves over the held-out horizon grid
        grid = report["band"]["grid"]
        errs = report["band"]["errors"]
        for i, curve in enumerate(errs):
            for tau, e in zip(grid, curve):
                out[f"band/err_f{i}_t{int(tau)}"] = float(e)
        # field-1-over-field-0 advantage curve + summary (the decisive signal)
        adv = report["band"].get("adv_1_over_0")
        if adv:
            for tau, a in zip(grid, adv):
                out[f"band/adv_t{int(tau)}"] = float(a)
            for key in ("adv_short_mean", "adv_long_mean", "separation"):
                if key in report["band"]:
                    out[f"band/{key}"] = float(report["band"][key])
            out["band/crossing"] = float(bool(report["band"].get("crossing", False)))
            if "field1_wins_frac" in report["band"]:
                out["band/field1_wins_frac"] = float(report["band"]["field1_wins_frac"])
            for key in report["band"]:
                if key.startswith("adv_at_tau"):
                    out[f"band/{key}"] = float(report["band"][key])
    if report.get("band_z"):
        for key in ("crossing", "separation", "adv_short_mean", "adv_long_mean", "field1_wins_frac"):
            if key in report["band_z"]:
                out[f"bandz/{key}"] = float(report["band_z"][key])
        out["bandz/crossing"] = float(bool(report["band_z"].get("crossing", False)))
        gridz = report["band_z"].get("grid", [])
        errsz = report["band_z"].get("errors", [])
        for i, curve in enumerate(errsz):
            for tau, e in zip(gridz, curve):
                out[f"bandz/err_f{i}_t{int(tau)}"] = float(e)
        advz = report["band_z"].get("adv_1_over_0")
        if advz:
            for tau, a in zip(gridz, advz):
                out[f"bandz/adv_t{int(tau)}"] = float(a)
        for key in report["band_z"]:
            if key.startswith("adv_at_tau"):
                out[f"bandz/{key}"] = float(report["band_z"][key])

    if report.get("speed"):
        for i, s in enumerate(report["speed"]["speed0"]):
            out[f"speed/field{i}"] = float(s)
        for i, s in enumerate(report["speed"]["straightness"]):
            out[f"straight/{i}"] = float(s)
    if report.get("composition"):
        out["compose/gap"] = float(report["composition"]["gap"])
    return out

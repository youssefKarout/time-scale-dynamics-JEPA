# Latent Time-Scale Dynamics JEPA (LTS-D-JEPA)

**Status: negative result.** Per-horizon velocity fields collapse to a single
rescaled flow on MOVi-A (gauge degeneracy); see
FINDINGS_movi_a_no_specialization.md.

Self-supervised video representation learning from **latent temporal dynamics**.
A shared pair-encoder maps two consecutive frames to a latent state `h0`; several
**per-horizon velocity fields** transport the *same* `h0` to different future
horizons via an ODE; an EMA target encoder provides the targets. The PoC asks a
single question:

> Does forcing different latent dynamics to predict different horizons, from the
> same `h0` and with **no semantic supervision**, spontaneously produce different
> temporal scales / persistence properties?

This repo is the **pure-emergent** study: the primary run uses only the horizon
objective plus target standardization (M5). VICReg variance/covariance is
anti-collapse only. Everything else (capacity ladder M2, per-branch EMA M4,
frequency reg M3, branch decorrelation M6) is deferred to controlled ablations.

## Method

```
h0 = E_theta(X_{t-1}, X_t)                                  # shared encoder
dh_k/dtau = V_{phi_k}(h_k, tau),  h_k(0) = h0               # per-branch field
h_k_hat = h0 + \int_0^{tau_k} V_{phi_k}(h_k(s), s) ds       # integrated (RK4, u in [0,1])
h_k* = sg[ E_xi(X_{t+tau_k-1}, X_{t+tau_k}) ]               # EMA target
L = sum_k w_k * D(h_k_hat, h_k*) + lambda * VICReg          # objective
xi <- beta * xi + (1 - beta) * theta                        # EMA
```

## Measurement suite (label-free)

| Def | Metric | Meaning |
|-----|--------|---------|
| 1 | speed / straightness | `||V(h0,0)||`, path length, `displacement/path` |
| 2 | **cross-horizon transfer gap (CHTG)** | does field i predict horizon j? `delta = offdiag - diag`; `0` ⇒ shared flow |
| 3 | **temporal receptive band** | per-field error over a held-out horizon grid `1..tau_max`; is `argmin` ordered by `tau_k`? |
| 4 | composition | short field re-applied to reach `tau_2` vs the long field |

## Controls (same code, `run.variant`)

| variant | role |
|---------|------|
| `primary` | the method (separate fields) |
| `shared`  | B1: one field integrated to both horizons |
| `horizon_cond` | B2: `g(h0, tau)` — classic multi-horizon JEPA |
| `regressor` | B3: independent MLP per horizon, no dynamics |

## Layout

```
configs/movi_a_poc.yaml        # single source of truth for the PoC
src/data/                      # MOVi-A loader + augmentations
src/models/                    # encoder, velocity field, integrator, EMA, standardizer, model
src/losses.py                  # JEPA distance + VICReg
src/metrics/specialization.py  # Def 1-4 + report/summarize
src/train.py                   # training/eval loop
src/eval/downstream.py         # frozen-feature probes vs MOVi-A GT (P3)
scripts/export_gt_trajectories.py
tests/smoke_test.py
```

## Setup

```
conda activate jepa_v
# local (no GPU):
pip install torch --index-url https://download.pytorch.org/whl/cpu
# colab (GPU):
# pip install torch
pip install -r requirements.txt
```

## Run

```
python tests/smoke_test.py                                   # CPU, fast validation
python -m src.train --config configs/movi_a_poc.yaml --smoke # tiny run -> runs/movi_a_poc_smoke/
python -m src.train --config configs/movi_a_poc.yaml         # full run
python -m src.train --config configs/movi_a_poc.yaml --set run.variant=shared   # control B1
```

> `--smoke` appends `_smoke` to `run.name`, so it writes to
> `runs/<name>_smoke/<variant>/` and **cannot** overwrite a real run's
> checkpoints or `train_log.csv`.

Data: per-clip `.npz` with `video` `[T=24, 128, 128, 3]` uint8 (default path in the config).

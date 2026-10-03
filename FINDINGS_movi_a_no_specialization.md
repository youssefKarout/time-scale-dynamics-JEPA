# FINDINGS — MOVi-A: no temporal-scale specialisation (a gauge degeneracy)

**Status:** PoC result on the `primary` variant, epoch 99, MOVi-A, `tau = [4, 16]`.
**One-line answer to the PoC question:** *No.* The two per-horizon velocity fields do
**not** specialise; they collapse onto **one and the same flow**, and the method
reduces **exactly** to the shared-field control **B1** (`run.variant=shared`).

---

## 1. What the standard metrics already said

| metric (raw) | value | reading |
|---|---|---|
| `chtg/diag` | 0.185 | fit is good |
| `chtg/offdiag` | 0.327 | off-diagonal **worse** than diagonal |
| `band/separation` | +0.206 | nonzero, but entirely from `adv_long` |
| `band/adv_short_mean` | +0.020 | field-1 already ≥ field-0 at short τ |
| `band/field1_wins_frac` | **1.0** | field-1 wins **every** horizon incl. τ=4 |
| `band/adv_short_mean` (z) | +0.0004 | the only "crossing" is a τ∈{1,2} tie of −0.0008 |

`crossing(raw)=False`. The apparent `crossing(z)=True` at τ ≤ 2 is an **artifact of
per-branch z-scoring** (see §4).

## 2. The decisive evidence: the two fields are one flow, rescaled

`scripts/diagnose_field_degeneracy.py` (checks A–C), 40 clips × 4 anchors:

```
(A) inter-field alignment at the anchor
    |h0| = 7.583   (h0 per-dim std = 0.0016)
    u=0.0: cos(v0,v1)=+0.9609  ||v0||=1.0967  ||v1||=0.3686  ratio=2.976
    u=1.0: cos(v0,v1)=+0.9301  ||v0||=1.1307  ||v1||=0.3291  ratio=3.436
    tau2/tau1 = 4.000

(B) gauge test:  cos / rel-dist  of  f0@(tau/s)  vs  f1@tau
     tau         s=0.25         s=1.0         s=4.0        s=16.0
       1   +0.983/ 0.63   +0.999/ 0.10   +1.000/ 0.01   +1.000/ 0.03
       2   +0.950/ 1.54   +0.997/ 0.21   +1.000/ 0.02   +1.000/ 0.06
       4   +0.835/ 4.77   +0.990/ 0.45   +1.000/ 0.04   +0.999/ 0.12
       8   +0.360/37.28   +0.969/ 1.01   +0.998/ 0.08   +0.995/ 0.23
      16   -0.919/75806   +0.879/ 2.62   +0.995/ 0.16   +0.986/ 0.42
      20   -0.870/3e7     +0.800/ 3.87   +0.992/ 0.20   +0.981/ 0.49
    best scale s* = 4.0 for EVERY tau (cos = 1.0000 at tau=1, 0.9996 at tau=4)

(C) rollout displacement  ||h(tau)-h0||/||h0||
    f0: 1: 0.144 4: 0.692  8: 1.743 16: 5.692 20: 9.508   <- diverges
    f1: 1: 0.041 4: 0.176  8: 0.386 16: 0.923 20: 1.261   <- bounded
```

**Reading.** The two fields are directional near-copies (`cos(v0,v1) = 0.96`,
`||v0||/||v1|| ≈ 3.1 ≈ tau2/tau1`), and their **integrated flows coincide under the
single substitution `tau -> tau/4`**: `f0` run to `tau/4` reproduces `f1` run to
`tau` with **cosine 0.99–1.000** and 1–20 % relative error across τ=1..20. `s* = 4`
is selected for *every* τ, exactly matching `tau2/tau1`. In other words the two
"separate" fields are the *same* dynamics law, with the entire branch rate ratio
absorbed by the external multiplier — i.e. the B1 control.

The training log agrees: `speed/field0 : speed/field1` converges to **~3.1** (from
1.0 at init) and stays there — the two field *magnitudes* separate by the horizon
ratio while their *directions* stay aligned.

## 3. Root cause: the horizon never reaches the network

The ODE is reparametrised to **normalised time `u ∈ [0,1]`**
(`src/models/integrator.py`: `dh/du = tau_k * V(h, u)`), and `VelocityField`
conditions only on `u` (`src/models/velocity_field.py` → `TimeEmbedding`).
Therefore **both fields receive byte-identical inputs `(h, u)` on every call**;
the physical horizon `tau_k` lives *only* in the external multiplier.

Consequences:

1. The objective is invariant under the **gauge transformation**
   `V <- (1/a)V, tau <- a*tau`. "Different temporal scale per branch" is
   therefore **not an identifiable target** — it is a redundant direction.
2. The easiest point in that gauge orbit is `V_phi0 = V_phi1` with the multiplier
   doing the work ⇒ exactly the B1 control.
3. Epoch sweep (check E) shows this *from epoch 0*. At τ=4 the gauge identity
   `cos(f0@τ/4, f1@τ)` is **0.993 at init** and **0.9996 at epoch 99** (at τ=16:
   0.913 → 0.995) — the collapse is **architectural**, not an optimisation failure
   that more epochs / a better EMA would fix.

> This also means the module docstring claim that "the horizon is still explicit
> (`V` is reparametrised, still a function of physical tau)" is **false in
> practice**: `V` is a function of normalised `u` only.

## 4. Task-space confirmations (labels used downstream only)

Single ridge probe fit **once** at "now" (`h0` → per-object GT centres, 25.4 px fit
floor), then applied *unchanged* to each rollout (`check_D`). `adv` is
`err_f0 − err_f1`, so **positive ⇒ field-0 is worse**:

| τ | f0 (px) | f1 (px) | adv (f0 − f1) |
|---|---|---|---|
| 1 | 35.7 | 25.2 | +10.6 |
| 2 | 64.5 | 27.4 | +37.1 |
| 4* | 150.6 | 37.2 | +113.4 |
| 8 | 456.6 | 67.7 | +388.9 |
| 16* | 2031.0 | 156.0 | +1875.0 |
| 20 | 3749.2 | 217.2 | +3532.0 |

(field-0's own trained horizon is τ=4; field-1's is τ=16.)

- field-1 wins **everywhere**, monotonically; there is no τ at which field-0 is better.
- field-0 is an **unstable integrator** (36 px → 3749 px), so it loses *even on its
  own trained horizon* τ=4 (151 px vs field-1's 37 px).
- **Cosine-distance JEPA cannot punish this**: cosine is scale-invariant, so a
  runaway rollout can still keep a small loss.
- The earlier z-space "crossing at τ≤2" vanishes here: per-branch z-scoring divides
  field-0's error by field-0's own (inflated) target scale, manufacturing a tie.

The rollout video (`runs/movi_a_poc/rollout_vs_others.gif`) shows the same
qualitatively: `primary` and `shared_wide` trace plausible short arcs, while
`horizon_cond` (B2) has a **fully time-collapsed encoder**
(`||h0(F)-h0(2)|| = 0.0002` over F=2..15) and freezes.

## 5. Why MOVi-A cannot exhibit the effect

MOVi-A is generated by a **single time-homogeneous flow** (rigid, near-ballistic
object motion). The Bayes-optimal predictor therefore has the *same functional
form* at every τ — it is one flow observed for different durations. A method can
only be pushed to "specialise" if the optimal predictor *changes structure* with
the horizon. Deriving the degeneracy: if `Phi` is the true flow, branch k must
realise `Phi^{tau_k}`, i.e. a field `~ tau_k * log(Phi)`. Any two branches are then
*forced* to be scalar multiples of one another. **The collapse is the correct
answer, and B1 is the correct model.**

⇒ The lever is the **benchmark**, not the optimiser. Escalations that make
specialisation *attainable*:

1. **Non-stationary dynamics** (variable friction / "wait-then-move" schedules) —
   the short-horizon optimal predictor differs from the long-horizon one.
   Cheapest: derive from MOVi-A.
2. **Interaction / contact outcomes** — IntPhys, CLEVRER: long τ requires reasoning
   about collision/occlusion; short τ does not.
3. **Regime switching** (bouncing with variable speeds) — Moving-MNIST style.

And the **method must be de-gauged** before those benchmarks are informative: feed
the field the *physical* scale (e.g. `log tau_k`, or absolute `tau_k * u`) — then
re-run against B2, because scale-conditioning alone degenerates to B2.

## 6. Reproduce

```bash
# 1. the decisive diagnostic (checks A-E): gauge degeneracy + task-space band test
python scripts/diagnose_field_degeneracy.py

# 2. the existing latent-space metric suite (band/separation, chtg/*)
python scripts/eval_checkpoint.py --ckpt runs/movi_a_poc/primary/ckpt_epoch99.pt

# 3. qualitative rollout video (one anchor, tau=1..20, per-object decode)
python scripts/render_rollout_video.py
```

## 7. Fixes made along the way

- Earlier per-object videos looked *frozen* because the ridge probe was fit on `h0`
  and then applied to *rolled* latents, and because the target (scene centre of
  mass) is itself nearly static. Both are corrected in
  `scripts/render_rollout_video.py` (probe fit at "now", per-object targets, one
  calibration for all τ).
- The measured latent barely moves across anchors anyway:
  `||h0(F) - h0(2)|| = 0.004 → 0.032` for `primary` over F=2..15 (|h0| ≈ 7.6), and
  only `0.0002` for `horizon_cond` — B2's encoder is fully collapsed in time. So a
  *sliding-anchor* video cannot animate much; the fixed-anchor τ-rollout is the
  meaningful visualisation.
- Fitting a **per-τ** probe is tautological: each τ gets a free linear map, so all
  models report a flat ~17 px floor. Never use a per-τ probe for the band test.

## 8. Incident: `--smoke` clobbered two checkpoints (fixed)

While validating an end-to-end run I executed
`python -m src.train --config configs/movi_a_poc.yaml --smoke`. `out_dir` is
`runs/<name>/<variant>` and `--smoke` did **not** alter it, so the tiny 3-epoch
run **overwrote** `runs/movi_a_poc/primary/ckpt_epoch0.pt`,
`ckpt_epoch1.pt` and `train_log.csv`.

Survived (untouched): `ckpt_epoch2.pt` … `ckpt_epoch99.pt`, `eval_band.json`,
and every evidence log (`diag_perm.log`, `diag4.log`, `pivot.log`,
`band_task.log`), all timestamped before the incident. The quoted epoch-0
numbers above therefore come from the *intact* pre-incident run recorded in
`runs/movi_a_poc/diag4.log`, and remain valid as evidence — but the raw
`ckpt_epoch0/1.pt` are no longer the originals, so `check_E` (and any per-epoch
sweep) must not be re-run expecting the pre-incident epoch-0 values.

**Fix applied** (`src/train.py`): `--smoke` now appends `_smoke` to `run.name`,
so it writes to `runs/<name>_smoke/<variant>/` and cannot overwrite a real run.
`train_log.csv` for `primary` is permanently lost (the full 100-epoch curve);
its key columns are quoted verbatim in §2 and in `runs/movi_a_poc/pivot.log`.



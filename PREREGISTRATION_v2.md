# VIBES pre-registration, version 2

Version 1 (`PREREGISTRATION.md`, commit `6a44689`) stays in the repository unchanged except for its
Deviations log. This document replaces it for all main experiments. It is committed **before any v2
main run**. Pilots on train/val pools (and on in-memory generated data) are allowed and listed in
Section 9. Every later change goes to the dated Deviations section at the end.

## 0. Why there is a version 2 (honest account)

The v1 world models never became competent, so the v1 questions could not be asked of them.
- Seed-0 REC/JEPA/STATE models (8k steps, batch 16, 16k episodes) blurred to the mean within about
  10 frames.
- Their belief states did not encode gravity: linear-probe R^2 for `g` was about 0.
- In free flight their predictions were no better than a constant-velocity baseline.

We found this by running the full v1 evaluation (including test/gsweep) on the seed-0 models. That
broke the v1 pilot clause, which forbade touching evaluation splits in the pilot. Nothing from that
evaluation is reported as a result. v2 uses freshly generated splits with new seeds, so no v2
evaluation episode was ever seen.

Pilot diagnostics on train/val only (scripts `scripts/proxy.py`, `scripts/pilot.py`) identified
three causes:
1. **Too little data.** With 16k episodes the training loss kept falling while validation error
   stalled: the models memorized episodes. A 200k pool removed the gap.
2. **Hidden restitution and friction.** These dominate long-horizon error (7-9 px over 24 frames,
   versus 3-5 px for not knowing `g`), so the loss rewarded learning contacts long before gravity.
3. **Training scheme.** Autoregressive open-loop training of small GRUs was slow and hedged toward
   the mean. Chunk prediction trained about 7x faster per step and learned `g` (proxy R^2 0.79
   after 40k steps).

The v1 `g_eff` estimator was also biased toward 0. It accepted the model's "soft bounces" (smooth
deceleration before a hedged contact) as free flight. v2 measures curvature only where the
*reference* trajectory is in exact free flight (Section 3).

## 1. World v2 (fixed)

- Same simulator and objects as v1: unit box, periodic x, floor and ceiling; 1-3 balls or blocks;
  mass = area. Per episode: `g ~ U[4, 8]`; `e = 1` w.p. 1/2 else `U[0.6, 1)`; `mu = 0` w.p. 1/2
  else `U[0.05, 0.4]`.
- **20 fps** (`dt = 0.05`). Episodes have 40 frames. Models see **C = 10** context frames and
  predict **R = 20** frames (1.0 s).
- **Visible materials.** The background tint encodes `e` (red) and `mu` (green). Gravity is not
  visible in any single frame and must be inferred from motion.
- Splits (`vibes/world.py`, seeds 20-30): train 100k, val 2k, test 2k, gsweep
  (`g` in {1, 1.5, ..., 12}, 100 per value), wide (train with `g ~ U[1, 12]`, 100k) and
  wide_val, long (conservative, 90 frames), count45, speed, massratio, mass_ctrl (1k each).

## 2. Model families (two-stage)

**Stage 1, perception.** A per-frame encoder (CNN, 64-d latent), trained on frames of the train
pool and then frozen.
- **REC:** autoencoder with foreground-weighted pixel MSE.
- **JEPA:** the same encoder trained to predict the EMA-target latent of frame t+1 from the latents
  of frames t-1 and t, with VICReg variance/covariance regularization and no decoder.
- **STATE:** no perception. Its features are ground-truth object positions, sizes and shapes plus
  the visible `e` and `mu` (no velocities, no `g`).

**Stage 2, dynamics (identical for all families).**
- A chunk world model, `vibes/dynamics.py`. The belief `h_C` is a GRU over the 10 standardized
  context latents and their first differences.
- A decoder MLP(`h_C`, k) predicts the 20 future latents as offsets from the last context latent.
- Loss: MSE in standardized latent space.
- Sizes S/M/L: belief width 128/256/512, decoder width 256/512/1024.
- Core ablation: diagonal linear recurrence (LRU) instead of the GRU.

**Readout.**
- STATE: positions are read directly from the predicted features.
- REC/JEPA: a **post-hoc pixel decoder**, trained identically for both families on frozen latents of
  real *val* frames, followed by soft colour-assignment centroids.
- REC's own training decoder is not used for readout, so both latent families are read out the
  same way.
- A colour-indexed MLP probe was tried first and rejected in the pilot: it reached only about 6 px
  on REC latents, whereas decoding the same latents gives 0.7 px.

## 3. Measuring gravity

`g_eff`:
- For each object, take every maximal run of at least 5 frames in which the **reference**
  trajectory is in exact free flight. The reference is the true or counterfactual simulator
  trajectory; a frame qualifies if the second differences around it equal `-g dt^2` with a
  1-frame margin.
- Fit a quadratic over each run that starts within the first 8 predicted frames.
- Combine runs by a weighted median, with weights equal to the curvature precision.
- Validation: exact on ground truth (max error 4e-6); per-episode sd 0.7 for 1 px i.i.d. noise.

Response-curve statistics, as in v1:
- in-range slope `s` (OLS over `g` in [4, 8]);
- `rho_hi = (gbar(11.75) - gbar(7.75)) / (4 s)` and `rho_lo = (gbar(4.25) - gbar(1.25)) / (3 s)`,
  where boundary values average the two nearest grid points.
- 1 = law, 0 = clamp.

## 4. Experiments

- **E0 competence gate.**
  - A family enters the OOD analyses only if its in-range natural slope on **val** is at least 0.5.
  - Families that fail are reported as failures, with their numbers.
- **E1 decodability.**
  - Ridge probes on `h_C` (fit on val, scored on test) for `g`, `e`, `mu`, `E`, `p_x`.
  - Compared with a model whose dynamics weights are random and untrained (same perception).
- **E2 natural response.**
  - `g_eff(g)` on gsweep.
  - Bayes-optimal reference: the truncated-normal posterior mean under `U[4, 8]` given context
    positions with 1 px Gaussian noise (pre-registered), plus a 0.25 px sensitivity curve.
- **E3 locus of failure.**
  - (a) *Internal estimate:* the ridge `g`-probe applied to `h_C` on gsweep gives `rho_inf`.
  - (b) *Linear write:* steering along `u_g`. `u_g` is estimated from 800 matched-state
    counterfactual pairs (time-reversed contexts ending in the same state under two in-range
    gravities). Targets `g*` in {1, ..., 12}. Reported quantities: `rho_exe`, `s_exe`, CER against
    the simulator counterfactual, and specificity.
  - (c) *Nested inversion* on 30 gsweep episodes per `g` in {1, 2, 3, 4, 6, 8, 9, 10, 11, 12},
    plus an **anti-physics control**: in-range contexts whose future is re-simulated with `g = -4`.
    - The objective is MSE on the first 10 predicted frames only (the fit window). All metrics use
      the **held-out** frames 10-19.
    - `inv_h` optimizes the belief within the GRU's range (`h = tanh(v)`) and tests execution.
    - `inv_z` optimizes the 10 context latents through the frozen belief GRU and tests
      reachability.
    - `nat` is the natural model.
  - (d) *Perception check:* position-readout error on gsweep context frames, out-of-range vs
    in-range `g`.
- **E4** (exploratory) velocity edits along a matched-pair `vx` direction: CER, and whether `p_x`
  and `E` change as the simulator says.
- **E5** conservation drift over 80 frames by chunked autoregression on `long`.
- **E6** (exploratory) count45 / speed / massratio position error.
- **E7 scaling.**
  - Data N in {1k, 8k, 100k} (nested prefixes of train).
  - Dynamics size S/M/L.
  - WIDE control trained on `g ~ U[1, 12]`.
  - LRU core ablation.

## 5. Hypotheses (confirmatory; one-sided hierarchical-bootstrap tests, Holm over all)

- **H1.** For REC, JEPA and STATE (M, 100k), the trained `g`-probe R^2 is at least 0.5 and greater
  than untrained. (3 tests)
- **H2 (Prop. 2).** Natural `rho_hi < 0.5` and `rho_lo < 0.5` for each family that passes E0.
  (up to 6 tests)
- **H3 (steering validity).** `s_exe >= 0.5` and in-range CER > 0. (3 tests)
- **H4 (scale does not buy extrapolation).**
  - The upper 95% CI of the slope of natural `rho_hi` against log10(N) is below 0.15 per decade.
  - In-range test error decreases with N. (3 tests)
- **H5 (perception is not the locus).** For REC and JEPA, the mean position-readout error on
  context frames at out-of-range `g` exceeds the in-range error by less than 0.5 px (upper 95% CI).
  Frames are individually in-distribution. (2 tests)

## 6. Locus decision rule (per family and side; reported with CIs, not a test)

- **Execution-limited:** `rho(inv_h)` < 0.5 with its CI below 0.5.
- **Inference-limited:** `rho(inv_h)` >= 0.5, `rho(inv_z)` >= 0.5, and CI of `rho(inv_z) - rho(nat)`
  above 0.
- **Unreachable belief:** `rho(inv_h)` >= 0.5 but `rho(inv_z)` < 0.5.
- **Otherwise:** mixed.

Interpretation guard: if `inv_h` also reproduces the anti-physics target
(`g_eff_hold` for `g = -4` below 0 on average), the decoder is flexible enough to produce non-laws.
"Execution-capable" then means only that, not "executes the law".

Here `rho` for an inversion mode uses `g_eff` on the held-out frames at the inversion targets:
- in-range slope `s` is the OLS slope over targets {4, 6, 8};
- `rho_hi = (mean_{11,12} - gbar(8)) / (3.5 s)`;
- `rho_lo = (gbar(4) - mean_{1,2}) / (2.5 s)`.

The same formula applied to `nat` on these episodes gives the paired natural value used in the rule.

## 7. Seeds, runs, statistics

- Main runs: REC, JEPA and STATE at M size on 100k, seeds {0, 1, 2}. For REC/JEPA the perception
  seed equals the dynamics seed, so each run is an independent replicate of both stages.
- Scaling cells, WIDE and LRU: seeds {0, 1}. They reuse the seed-matched perception.
- The unit of replication is the training run.
- Statistics: hierarchical bootstrap (seeds, then episodes; 2,000 resamples, percentile CIs),
  one-sided p-values, Holm-Bonferroni over all confirmatory tests at alpha = 0.05.
- With 2-3 seeds the seed-level variance is poorly estimated; we say so.

## 8. Theory (unchanged from v1)

- **Prop. 1:** competence implies decodability.
- **Prop. 2:** a Bayes-optimal squared-loss predictor clamps out-of-range parameters. Without
  observation noise, the loss does not constrain out-of-range contexts at all.

## 9. Pilots before this commit (train/val or in-memory data only)

`scripts/proxy.py` and `scripts/pilot.py` runs on generated pools. They compared:
- autoregressive vs chunk dynamics;
- 16k vs 200k episodes;
- hidden vs visible materials;
- dt 0.04 vs 0.05;
- MLP vs GRU belief.

Smoke tests of every pipeline used 3k-step models. Training steps for stage 1 and stage 2 are
fixed in the Deviations log below, from train/val loss and val competence only, before any main
run.

**Fixed before any main run (from train/val/pilot-split checks only).**
- Stage 1: 12,000 AdamW steps, batch 64 frame windows, lr 1e-3, cosine schedule.
- Stage 2: 60,000 steps, batch 128 windows, lr 1e-3, cosine schedule. The STATE seed-0 pilot
  plateaued by 20k-40k steps, with val competence slope 0.91.
- JEPA stage-1 regularization and prediction horizon are chosen by readout precision on a
  separate `pilot` split (seed 31). The baseline JEPA recipe reached only 4.9 px, against 0.7 px for
  REC. The choice is recorded in Deviations before any JEPA main dynamics run. REC uses its first
  recipe and is not tuned.

## Deviations

- 2026-11-06, disclosure. Before the `pilot` split existed, the per-frame readout precision of
  the seed-0 encoders was measured once on frames of 300 *test* episodes. This number is not part
  of any hypothesis. H5 uses gsweep context frames.

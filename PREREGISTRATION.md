# VIBES pre-registration

Committed on 2026-11-05, before any world model was trained. The simulator, renderer and their
tests already exist. The literature check that motivated the design is in `docs/LITERATURE.md`.
Every later change goes into the dated **Deviations** section at the end. Nothing above it is
edited after this commit.

## 0. Question

When a video world model that infers a physical parameter (gravity `g`) from context frames fails
outside the training range, *where* does it fail?

- **Inference failure.** The belief state never represents an out-of-range `g`.
- **Execution failure.** The learned transition cannot carry out an out-of-range `g`, even when the
  belief state is set to one.

What *should* an ideal learner trained with the same loss do? (Section 6 gives the theory.) And
does any of this depend on the training signal (pixel reconstruction, JEPA latent prediction,
ground-truth state) or on scale?

## 1. World and data (fixed)

- 2D unit box, periodic in x, floor at y=0, ceiling at y=1. Balls (radius 0.06-0.10) and
  non-rotating square blocks (half-size 0.05-0.085, P(block)=0.3). Mass = area. 1-3 objects.
- Per episode: `g ~ U[4, 8]`; `e = 1` w.p. 0.5 else `U[0.6, 1)`; `mu = 0` w.p. 0.5 else
  `U[0.05, 0.4]` (floor/ceiling friction). Positions `y ~ U[0.25, 0.85]`, velocities
  `vx, vy ~ U[-1.5, 1.5]`.
- 25 fps (`dt = 0.04`), 40 frames per training episode, 64x64 RGB rendered on the GPU. Ground-truth
  state, energy `E` and momentum `(p_x, p_y)` are saved for every frame.
- Splits and seeds (`vibes/data.py`): train 16,000 episodes (seed 0); val 1,000 (seed 1; used
  only to fit probes and directions); test 2,000 (seed 2); `gsweep`: `g` in {1, 1.5, ..., 12},
  2,300 episodes (seed 3); `long`: conservative (`e=1, mu=0`), 72 frames, 1,000 episodes (seed 4);
  `count45`: 4-5 objects (seed 5); `speed`: speed in [2.2, 4.2] (seed 6); `massratio`: two balls
  with radii 0.04 and 0.14, mass ratio 12.25 (seed 7) with its control `mass_ctrl` (seed 8);
  1,000 episodes each.

## 2. Models (fixed architecture family; training hyperparameters may be adjusted in the pilot only)

All models share the same recurrent core: a per-frame encoder produces `z_t` (64-d). A GRU
(hidden size `H`) consumes `z_t` for `C = 8` context frames, then runs open-loop on its own
predictions `z_hat` for the remaining frames. This is the Dreamer-style deterministic
"imagination" setup.

- **REC** (pixel reconstruction): CNN encoder plus deconvolutional decoder. Loss: pixel MSE of
  decoded predictions plus an autoencoding term.
- **JEPA** (latent prediction): the same CNN encoder and no decoder. Targets come from an EMA target
  encoder. Loss: normalized latent MSE plus VICReg variance/covariance regularization.
- **STATE** (reference): the same GRU core on ground-truth object positions and sizes (no
  velocities, no parameters), predicting next positions. It has perfect perception and the same
  inference problem.
- Sizes: S/M/L = CNN width 16/32/64, GRU `H` = 128/256/512.
- Training: random 32-frame windows, 8 context and 24 open-loop frames, AdamW. The step count is
  fixed per configuration after the pilot.

**Runs.** The main configurations are REC-M, JEPA-M and STATE-M on 16k episodes with seeds
{0, 1, 2}. Scaling uses 2 seeds (0, 1) each: REC and JEPA at sizes S and L on 16k, and at size M on
1k and 4k episodes. That makes 22 pixel runs and 3 state runs. Untrained (random-init) copies serve
as probe baselines.

**Pilot.** At most 3 short runs on train/val only, to set learning rate, step count, batch size
and loss weights, and to check that readouts work. No confirmatory metric is computed on
test/gsweep/OOD splits during the pilot.

## 3. Readouts (how physics is measured from a model)

- **Primary: decoded-pixel readout.** Predicted latents are decoded to frames (REC: its own
  decoder; JEPA: a decoder trained *post hoc* on frozen target-encoder latents of real val frames;
  STATE: positions directly). Each object's position is the soft color-assignment centroid, using
  a circular mean in x. Velocities are central differences.
- **Secondary: latent readout.** An MLP probe from per-frame latents to object positions, fit on
  real val frames.
- **Effective gravity `g_eff`.** Fit a quadratic to every 7-frame window of an object's predicted
  track in which the object is in free flight. Free flight means its readout position is at least
  `size + 0.03` from the floor and ceiling and it is not near another object. Then
  `g_eff = median(-2 * curvature)` per episode. Episodes with no valid window are dropped and
  their count is reported. The estimator is validated on ground-truth frames (test in `tests/`).

## 4. Experiments, metrics and hypotheses

Notation: `h_C` is the GRU state after the 8th context frame.

**E1 Decodability.** Ridge probes `h_C -> g, e, mu, E, p_x` (fit on val, scored on test) for trained
and untrained models: R^2.
- **H1 (confirmatory).** For REC-M and JEPA-M, the linear probe R^2 for `g` is >= 0.8 and is larger
  than for the untrained model.

**E2 Natural OOD response.** On `gsweep`, open-loop rollouts give `g_eff(g_true)`.
- In-range slope `s_ID`: OLS slope over `g_true` in [4, 8].
- Out-of-range slopes, normalized: `rho_hi = (gbar_eff(12) - gbar_eff(8)) / (4 * s_ID)` and
  `rho_lo = (gbar_eff(4) - gbar_eff(1)) / (3 * s_ID)`. Here `gbar` is the mean over episodes with
  that `g_true`, and boundary values average the two grid points nearest the end (e.g. 11.5 and 12).
  The law gives `rho = 1`; clamping gives `rho = 0`.
- Bayes reference: the posterior mean of `g` under the training prior `U[4, 8]` given the context
  positions with Gaussian noise of 1 px (closed form, Section 6).
- **H2 (confirmatory, from Prop. 2).** For REC-M and JEPA-M, `rho_hi < 0.5` and `rho_lo < 0.5`:
  OOD gravity is closer to the clamp than to the law.

**E3 Locus of failure.**
- (a) **Internal estimate.** The ridge `g`-probe applied to `h_C` on `gsweep` gives
  `g_probe(g_true)` and its normalized out-of-range slope `rho_inf` (same formula as above, using
  the probe's in-range slope).
- (b) **Execution.** An encoding direction `u_g` is estimated from matched-state counterfactual
  pairs. For conservative val episodes, we construct contexts that end in exactly the same state
  under two gravities `g, g'` in [4, 8] by time-reversed simulation, and set
  `u_g = mean[(h_C(g') - h_C(g)) / (g' - g)]`. On ID test episodes with `g` in [5, 7], we steer
  `h_C <- h_C + delta * u_g` so that the probe reads a target `g*` in {1, 1.5, ..., 12}, roll
  out, and measure `g_eff`. This gives the execution slopes `rho_exe_hi` and `rho_exe_lo`
  (normalized by the steered in-range slope) and the in-range transfer slope `s_exe`.
- Specificity: the readout position at `C+1` must stay within 1 px of the unsteered rollout.
- Counterfactual error reduction for in-range targets:
  `CER = 1 - err(steered, sim_from_state(g*)) / err(unsteered, sim_from_state(g*))`.
- **Decision rule (confirmatory).**
  - *Inference-limited*: the bootstrap CI of `rho_exe_hi - rho_hi` is above 0 and `rho_exe_hi >= 0.5`.
  - *Execution-limited*: `rho_exe_hi < 0.5` with its CI below 0.5.
  - Otherwise: *mixed/undetermined*.
  - The same rule is applied separately to the low side and to each family. We make no
    directional prediction about which locus wins.
- **H3 (confirmatory).** The steering is valid: `s_exe >= 0.5` and `CER > 0` for in-range targets
  (REC-M, JEPA-M). If H3 fails for a family, its locus classification is reported as
  "not identifiable".

**E4 Velocity edits and conservation bookkeeping (secondary, exploratory).** Encoding directions
for `vx` of one object come from matched-state pairs (time reversal from the edited state). We
steer, then compare to the simulator counterfactual (`CER`) and measure whether the momentum change
`m*dvx` persists through later collisions (`mu = 0`) and the energy change persists
(conservative episodes). Persistence is the ratio of the model's change to the true change.

**E5 Conservation drift (secondary).** On `long`, roll out 64 frames. Measure relative energy
drift `|E_hat(t) - E(t)| / E(t)` and momentum drift at `t = 8, 16, 32, 64` from the readout, for all
three families. We also check whether models dissipate correctly: correlation between model and
true `E(end) / E(C)` on test episodes with `e < 1`.

**E6 Other OOD (exploratory).** Rollout position error on `count45`, `speed` and `massratio`,
relative to ID test or `mass_ctrl`.

**E7 Scaling.** For REC and JEPA, `rho_hi`, `rho_lo`, `rho_exe_hi`, ID rollout error and `g`-probe
R^2 as functions of data (1k/4k/16k) and size (S/M/L). The slope of each metric is estimated against
log10(N) and log10(params).
- **H4 (confirmatory, a Kang et al.-style prediction).** ID rollout error decreases with data.
  Extrapolation does not meaningfully improve: the upper 95% CI of the slope of `rho_hi` against
  log10(N) is < 0.15 per decade.

## 5. Statistics

- The unit of replication is the training run (seed). We report per-run values, the mean over
  seeds, and hierarchical bootstrap 95% CIs (resample seeds, then episodes within seeds; 2,000
  resamples, percentile intervals).
- One-sided bootstrap p-values for H1 (two families), H2 (two families x two sides), H3 (two
  families) and H4 (two families). Holm-Bonferroni correction over these 10 tests at alpha = 0.05.
  The locus classification (E3) follows its decision rule with 95% CIs and is reported as is.
- With 2-3 seeds, seed-level variation is estimated poorly. We say this explicitly and do not make
  claims that need a seed-level test.
- Everything secondary or exploratory is labelled as such. All outcomes are reported, positive or
  negative.

## 6. Theory to be written up (stated before the experiments)

- **Prop. 1 (competence implies decodability).** If a predictor's multi-step prediction error is
  at most `eps` and the future-observation map separates values of a quantity `Q` with modulus
  `kappa` (`||F(s, Q1) - F(s, Q2)|| >= kappa |Q1 - Q2|`), then `Q` is decodable from the
  predictor's state with error `<= 2 eps / kappa`. So a positive probe follows from low loss alone
  and is weak evidence of "physics".
- **Prop. 2 (Bayes-optimal clamping).** Under squared loss and a prior on `g` supported on
  `[a, b]`, the implied gravity of the Bayes-optimal free-flight prediction is the posterior mean
  `E[g | context]`, which lies in `[a, b]` for any context. With Gaussian observation noise it
  converges to `clamp(g*, a, b)` as context grows. Without noise, the objective does not constrain
  OOD contexts at all. Either way, extrapolating the law must come from inductive bias, not
  from the loss.

## Deviations

(none yet)

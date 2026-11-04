# Literature and novelty check (done 2026-11-05, before any experiment)

The question "do world models learn physics or memorize pixels?" is crowded. This file records
what is already known, which candidate angles turned out to be answered, and the angle we chose.
Every arXiv id below was checked against the arXiv API (`scripts/verify_refs.py`, log in
`docs/refs_verified.txt`).

## 1. What is already known

**Behavioural tests of video models.**
- Kang et al. (2411.02385, ICML 2025) train diffusion video models on a 2D simulator (uniform motion,
  collisions, parabolas). They find near-perfect in-distribution behaviour, gains on combinatorial
  generalization with scale, and failure out of distribution (OOD). Generalization is "case-based":
  the model copies the closest training example, matching color before size, velocity and shape.
  The paper is purely behavioural and has no analysis of internal states.
- Physics-IQ (Motamed et al., 2501.09038, WACV 2026). Real-world physics benchmark; physical
  understanding of large video generators is severely limited and unrelated to visual realism.
- Morpheus (Zhang et al., 2504.02918) scores generated videos against conservation laws.
- Thozhiyoor et al. (2512.02016) measure the effective gravity of generated falling objects
  (sub-Earth gravity, violations of Galileo's principle).
- Learning How the World Evolves (Li et al., 2608.09926) adds an explicit kinematic-integration
  bias and reports a much smaller ID-to-OOD gap on PhyWorld-style tasks.

**Intuitive physics and violation of expectation.**
- IntPhys (Riochet et al., 1803.07616); PLATO (Piloto et al. 2022, Nature Human Behaviour,
  doi:10.1038/s41562-022-01394-8); Physion (Bear et al., 2106.08261).
- V-JEPA (Bardes et al., 2404.08471) and Garrido et al. (2502.11831): V-JEPA trained on natural
  video detects IntPhys violations (98% zero-shot) using a surprise metric; pixel-prediction
  models are close to chance.

**Probing and steering inside video and world models.**
- Joseph et al., "Interpreting Physics in Video World Models" (2602.07050): a "Physics Emergence
  Zone" at intermediate depth of large video encoders; direction is a distributed circular code.
- Alam (2605.24322, CVPR 2026 workshop): concept-activation-vector steering in VideoMAE shifts
  IntPhys plausibility judgments. It does not involve rollouts or continuous physical parameters.
- Tan et al. (2607.27017, PokeWorld): which physical parameters (mass, drag, stiffness) are
  decodable from latent world-model states depends on the prediction targets and horizon. They
  note that decodability and forecast dependence can come apart.
- Liu and Chen (2606.28455, 2608.15156, 2609.21787): event-conditioned diagnostics and low-rank
  "carrier" interventions for counterfactual velocity edits in recurrent world models on a
  two-object collision environment.
- RL world models: probing IRIS/DIAMOND (2603.21546).

**Conserved quantities: decodable vs used (this candidate is largely answered).**
- Liao and Cao, "The Objective Decides" (2607.03728): on state-input MLP/GRU/Transformer
  dynamics models, energy is decodable at R^2 ~ 1 but causally inert for next-state prediction,
  unless the objective rewards it. Single-step interchange along a probe direction.
- Bao (2608.23526): DreamerV3 on 64x64 pendulum video. An approximately conserved latent scalar
  tracks energy (|rho| ~ 0.97); pushing rollouts back to its level set lowers the error by about 3%.
  Future work they name: contacts, multiple interacting objects.
- Wang (2606.24945): certified horizons for decoded physical invariants of latent world models.
- Interno et al. (2602.12218): frozen linear probes recover energy and inverse-square gravity;
  invasive adaptation corrupts them.
- LeHami (2610.15267): symplectic latent dynamics for JEPA world models.

**World representations and the probing debate.**
- Othello-GPT: Li et al. (2210.13382), Nanda et al. (2309.00941), Yuan and Sogaard (2503.04421).
- Vafa et al. (2406.03689): generative models can predict well with incoherent world models.
- Vafa et al. (2507.06952, ICML 2025): transformers trained on orbits do not acquire a Newtonian
  inductive bias. Liu et al., "From Kepler to Newton" (2602.06923): short context yields
  force-like (Newtonian) solutions.
- Probing methodology: control tasks (Hewitt and Liang, 1909.03368), amnesic probing (Elazar et
  al., 2006.00995), "probing the probing paradigm" (Ravichander et al., 2005.00719), causal
  abstraction and interchange interventions (Geiger et al., 2106.02997; DAS, 2303.02536),
  decoding weights are not encoding patterns (Haufe et al. 2014, NeuroImage,
  doi:10.1016/j.neuroimage.2013.10.067).

**Physics-structured and latent world models.**
- HNN (Greydanus et al., 1906.01563), LNN (Cranmer et al., 2003.04630), Hamiltonian Generative
  Networks from pixels (Toth et al., 1909.13789).
- Dreamer (Hafner et al., 1912.01603), DreamerV3 (2301.04104), World Models (Ha and
  Schmidhuber, 1803.10122). JEPA family: I-JEPA (2301.08243), V-JEPA 2 (2506.09985); VICReg
  (2105.04906) and BYOL (2006.07733) for collapse prevention.
- Contextual world models that infer physical parameters: cRSSM (Prasanna et al., 2403.10967),
  DALI (2508.20294), PhysiCalWorld (2610.30125). Lin and Sun (2610.29088): how a parameter enters
  the mechanism decides whether out-of-range values help.
- Extrapolation theory: ReLU MLPs extrapolate linearly along rays (Xu et al., 2009.11848).

## 2. Candidate angles and their status

| Candidate | Status | Why |
|---|---|---|
| Do latents linearly/causally encode energy and momentum? | **Mostly answered** | Liao and Cao (state models, all invariants decodable, causally inert under next-state loss); Bao (pixel pendulum DreamerV3); Interno et al. Remaining gap: contact-rich multi-object pixel scenes. We keep a small, secondary version of this (E5) and do not sell it as the main novelty. |
| OOD gravity: extrapolate the law or snap back? | **Behaviour known, mechanism open** | Kang et al. and others show OOD failure from the outputs. Nobody has asked *where* inside the model the failure happens, or what an ideal learner trained with the same loss *should* do. |
| Architecture / scale dependence | Partly known | Kang et al.: scale does not fix OOD (diffusion only). Garrido et al.: JEPA beats pixel prediction on VoE surprise. No study compares JEPA and reconstruction on interventional OOD tests in a controlled setting. |

## 3. What is precisely unknown (our questions)

1. **Locus of OOD failure.** When a world model that infers gravity in context fails on
   out-of-range gravity, is the failure in *inference* (the belief state never represents a
   gravity beyond the training range) or in *execution* (the transition cannot carry out an
   out-of-range gravity even when the belief state is set to one)? This requires interventions:
   read the model's internal gravity estimate (probe), and separately write out-of-range values into
   the belief state (steering along an encoding direction) and measure the gravity its rollout
   actually exhibits.
2. **What should an ideal learner do?** If the Bayes-optimal predictor for the training objective
   itself snaps back, then "snap back" is not a learning failure but the objective doing its job,
   and extrapolation can only come from inductive bias. Nobody has stated this for world models
   or used it as a quantitative reference curve.
3. **Does the training signal matter (JEPA vs pixel reconstruction vs ground-truth state), and does
   scale change any of this?**
4. Secondary: in contact-rich, multi-object pixel worlds, do velocity edits propagate through
   collisions with the right momentum and energy bookkeeping, and how fast does conservation drift?

## 4. Chosen angle

"Inference or execution?" A causal dissection of OOD failure in video world models that infer
physical parameters in context, with a Bayes-optimal reference derived in closed form. Compared
across pixel-reconstruction, JEPA and ground-truth-state models, over data and model scale.

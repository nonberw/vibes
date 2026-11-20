# VIBES: do video world models learn physics, or just vibes?

**A pre-registered, causal dissection of where world models fail to extrapolate a physical law. Mixed results, reported as they came out.**

[Paper (PDF)](paper/main.pdf) · [Pre-registration v2](PREREGISTRATION_v2.md) · [v1 (the one that failed)](PREREGISTRATION.md) · [Overview in Russian (PDF)](docs/vibes_overview_ru.pdf) · [Submission checklist](docs/SUBMISSION.md)

![where the failure lives](results/v2/fig_locus.png)

---

## The question

World models watch a few frames, guess the physics, and predict the future. In distribution they
look great. Push one physical parameter outside the training range (here: gravity) and they fall
apart. Plenty of people have shown *that* ([Kang et al.](https://arxiv.org/abs/2411.02385),
[Physics-IQ](https://arxiv.org/abs/2501.09038)). Far fewer have asked *where*.

So: when a model that infers gravity from context meets a gravity it has never seen, does it fail
because

1. its belief never represents that gravity (**inference**),
2. its dynamics cannot execute that gravity even if told (**execution**), or
3. it misreads the frames (**perception**)?

And is it even a failure? Spoiler from the theory: under squared loss the Bayes-optimal predictor's
implied gravity is the posterior mean, which lives inside the training range no matter what you
show it. Snapping back to the boundary is the loss doing its job. If a model extrapolates the law,
that came from an inductive bias, not from the objective.

## What we found

Short version: we went looking for the one place where it breaks, and the honest answer is "it's
complicated, and our pre-registered ruler was too wobbly to measure it". No rounding up. Receipts:

- **The pre-registration mostly did not confirm.** 4 of 17 confirmatory tests survive Holm
  correction. The pre-registered out-of-range estimator (curvature of predicted height) is too noisy
  above the range and biased below it, so it cannot tell "clamps to the boundary" from "follows the
  law", and the pre-registered locus rule spits out labels with no stable pattern. That is a design
  miss on our side and the paper says so in the limitations.
- **Perception is fine.** Out-of-range frames are read as accurately as in-range ones. Not the locus.
- **The belief tracks out-of-range gravity but under-reads it by about half.** This is the tightest
  number in the study and it holds for all three families. Plot twist: a probe on frozen V-JEPA 2
  features under-reads by a similar factor, so some of this may be geometry of the code and not
  "the model refuses to believe".
- **Models neither clamp nor follow the law** (exploratory metric). Above the training range the
  predictions go about half-way from clamp to law. Below it they stay near the clamp. The theory
  says a Bayes-optimal learner under noise clamps; these networks do something in between.
- **The failure is split between inference and execution** (exploratory). For the one family where
  writing into the belief is valid (STATE), writing the true gravity gets predictions above the
  range to the level of a twin model trained on the wide range. Below the range, even the best
  belief an optimizer can find closes only about half of the gap to that twin.
- **"Scale does not buy extrapolation" was our hypothesis and it was not supported.** In-range error
  drops about 2 px per decade of data, and the exploratory metric says more data bought some
  extrapolation too. A bigger dynamics network did not. A linear recurrence made it worse.
- **Nobody conserves energy.** Median relative energy error roughly doubles within 80 frames.
- **Version 1 of this study flopped** (models never learned physics at all). We kept the corpse in
  the repo and in Appendix A, because the autopsy is informative.

Numbers below are for the main configuration (M dynamics, 100k episodes), 95% hierarchical bootstrap
intervals in brackets. Rows marked *exploratory* were added after the first evaluations and are not
covered by the pre-registration.

<!-- RESULTS_TABLE_START -->
| | STATE (true positions) | REC (pixels) | JEPA (latents) |
|---|---|---|---|
| seeds | 3 | 3 | 3 |
| competence gate: in-range slope of g_eff vs g (need >= 0.5) | 0.87 | 0.83 | 0.79 |
| in-range 20-frame error, px | 6.3 | 10.5 | 9.3 |
| gravity probe R^2, trained / untrained dynamics | 0.59 / 0.08 | 0.49 / 0.04 | 0.52 / 0.04 |
| **belief reading of out-of-range g**, slope above / below range (1 = tracks it, 0 = clamps) | 0.50 [0.43, 0.56] / 0.74 [0.58, 0.87] | 0.45 [0.41, 0.50] / 0.65 [0.58, 0.73] | 0.45 [0.38, 0.52] / 0.65 [0.53, 0.77] |
| perception: extra position error on out-of-range frames, px | -- | 0.01 [-0.07, 0.09] | 0.09 [-0.11, 0.29] |
| steering validity: in-range transfer slope (need >= 0.5), CER | 0.69 [0.59, 0.80], 0.15 | 0.22 [-0.05, 0.49], 0.05 | 0.57 [0.34, 0.84], 0.07 |
| pre-registered natural rho_hi (1 = law, 0 = clamp) | 0.43 [0.25, 0.62] | -0.08 [-1.53, 0.93] | 0.64 [-0.22, 1.55] |
| pre-registered natural rho_lo (biased upward, see paper) | 1.28 [1.04, 1.55] | 1.63 [0.93, 3.49] | 1.86 [1.12, 3.54] |
| pre-registered locus label, above / below range | unreachable belief / mixed | mixed / mixed | mixed / mixed |
| *exploratory* law fraction, natural, above / below range | 0.55 [0.52, 0.58] / 0.28 [0.25, 0.31] | 0.70 [0.64, 0.76] / -0.16 [-0.22, -0.11] | 0.53 [0.49, 0.58] / 0.13 [0.08, 0.19] |
| *exploratory* law fraction after writing true g into the belief, above / below | 0.85 / 0.38 | 0.65 / 0.13 (write not valid) | 0.75 / 0.25 (write not valid) |
| *exploratory* held-out law fraction above range: natural -> optimized belief -> WIDE ceiling | 0.49 -> 0.54 -> 0.72 | 0.55 -> 0.44 -> 0.74 | 0.46 -> 0.42 -> 0.62 |
| *exploratory* held-out law fraction below range: natural -> optimized belief -> WIDE ceiling | 0.31 -> 0.50 -> 0.74 | 0.10 -> 0.28 -> 0.43 | 0.25 -> 0.45 -> 0.63 |
| *exploratory* share of the gap to WIDE closed by the optimized belief, above / below | 21% [3%, 42%] / 43% [35%, 52%] | -58% [-109%, -22%] / 54% [32%, 74%] | -23% [-64%, 12%] / 53% [41%, 63%] |
| anti-physics control: law fraction toward upward gravity under optimized belief | 0.35 | 0.31 | 0.33 |
| data scaling 1k / 8k / 100k episodes: in-range error, px | 10.8 / 8.6 / 6.3 | 14.6 / 11.8 / 10.3 | 13.6 / 11.3 / 9.3 |
| data scaling 1k / 8k / 100k: *exploratory* law fraction above range | 0.24 / 0.39 / 0.55 | 0.50 / 0.44 / 0.70 | 0.41 / 0.40 / 0.53 |
| energy drift (median relative error), frames 0-20 -> 60-80 | 0.17 -> 0.39 | 0.27 -> 0.43 | 0.19 -> 0.41 |

Pre-registered confirmatory tests surviving Holm correction: **4 of 17** (H1_state, H3_state, H5_jepa, H5_rec). Full table: [`results/v2/hypotheses.md`](results/v2/hypotheses.md).

Outside reference: a ridge probe on frozen V-JEPA 2 (ViT-L) features of the same context frames reads gravity at cross-validated R^2 0.47, and its out-of-range slope above the range is 0.58.
<!-- RESULTS_TABLE_END -->

Read the [paper](paper/main.pdf) for the full numbers with confidence intervals, the Holm-corrected
hypothesis table, and the limitations. Everything in the paper is generated from
`results/v2/summary.json` by `scripts/paper_tables.py`, so there is nothing to transcribe by hand
and nothing to fudge.

## How it works

**World.** A 2D box with 1-3 balls and blocks. Every episode draws its own gravity `g ~ U[4, 8]`,
restitution and friction. Materials are *visible* (the background tint encodes restitution and
friction); gravity is not visible in any single frame and has to be inferred from motion, which is
the honest version of the real world: rubber looks different from steel, `g` does not look like
anything. 20 fps, 64x64, 10 context frames, 20 predicted frames, 100k training episodes. The test
sweep goes `g = 1 ... 12`.

**Three families, one dynamics model.** A frame encoder is trained and frozen; a chunk world model
(GRU belief, MLP decoder for all 20 future latents at once) is trained on its latents. The families
differ only in the encoder's training signal:

- **REC**: pixel reconstruction (a keypoint autoencoder; the plain conv-AE never passed the competence gate).
- **JEPA**: latent prediction with an EMA target, VICReg regularized, no decoder.
- **STATE**: ground-truth positions, no velocities, no `g`. Perfect perception, same inference problem.

**Three gravity readings per model.**

1. *Natural*: what gravity the predictions exhibit (`g_eff`, curvature of predicted height, measured
   only where the reference trajectory is in free flight).
2. *Internal*: what a linear probe reads from the belief.
3. *Write*: steer the belief along a gravity direction and see what the dynamics execute. The
   direction comes from exact matched-state counterfactual pairs: conservative dynamics are
   time-reversible, so we can build two contexts that end in exactly the same state under two
   different gravities.

**Nested inversion.** A linear write can fail for boring reasons, so we also optimize the input of
the frozen model: the belief itself (tests execution), or the context latents fed to the frozen
belief network (tests reachability). We fit only the first 4 predicted frames and measure physics on
held-out frames. An anti-physics control (gravity pointing up) tells us how flexible the decoder is.

**Pre-registered.** Hypotheses, metrics, decision rules and tests were committed before any v2 model
was evaluated. Every deviation is dated and explained in `PREREGISTRATION_v2.md`. Version 1 of this
study (`PREREGISTRATION.md`) produced models that never learned physics; we kept it, because how
it failed is useful (Appendix A of the paper).

## Reproduce

```bash
pip install -r requirements.txt
```

```bash
python -m pytest tests -q
```

Generate every split (deterministic, about a minute):

```bash
python -c "import sys; sys.path.insert(0,'.'); from vibes import world; [world.ensure(k) for k in world.SPLITS]"
```

Run the whole grid (stage 1 encoders, stage 2 dynamics, evaluation; restartable; one GPU job at a time):

```bash
python scripts/run_v2.py --parallel 1
```

The per-model evaluation outputs of all 33 models are committed in `results/v2/runs/` (about 50 MB),
so the statistics, figures, tables and the paper can be rebuilt without training anything:

```bash
python scripts/final.py
```

That runs the aggregation (hierarchical bootstrap, Holm), regenerates every macro and table the
paper uses, refreshes the table in this README and compiles both PDFs. Set `VIBES_NB=200` for a
quick pass. The individual steps, if you want them one at a time:


```bash
python scripts/aggregate2.py
```

```bash
python scripts/paper_tables.py
```

```bash
cd paper && tectonic main.tex
```

One model by hand:

```bash
python scripts/train_perception.py --kind rec --enc kp --seed 0 --out runs/v2/perc_kp_rec_s0
```

```bash
python scripts/train_dynamics.py --family rec --perc runs/v2/perc_kp_rec_s0 --seed 0 --steps 30000
```

```bash
python scripts/evaluate2.py runs/v2/dyn_rec_M_gru_train_100k_s0
```

The outside reference on frozen V-JEPA 2 (downloads the ViT-L weights from Hugging Face, about 7 minutes on the same GPU):

```bash
python -P scripts/vjepa2_probe.py --device cuda --n_val 1000 --batch 8
```

Hardware used: one laptop GPU with 4 GB (RTX 3050 Ti). A stage-1 run is about 25 min, a stage-2 run
about 20 min, a full evaluation about 10 min.

## Layout

```
vibes/sim.py           exact-impact 2D physics (numba), energy/momentum bookkeeping
vibes/render.py        anti-aliased renderer with material tint (torch, GPU)
vibes/world.py         v2 splits, seeds, GPU pools
vibes/perception.py    stage 1: REC / JEPA frame encoders (conv, spatial-softmax, keypoint)
vibes/dynamics.py      stage 2: chunk world model (GRU or LRU belief + horizon decoder)
vibes/stage2.py        latent caches, readout decoder, stage-2 training
vibes/analysis2.py     uniform model interface, readouts, competence gate
vibes/experiments2.py  E1-E6: probes, natural sweep, Bayes reference, steering, nested inversion,
                       velocity edits, drift, OOD
vibes/readout.py       colour centroids and the reference-masked g_eff estimator
scripts/               grid runner, evaluation, aggregation (bootstrap + Holm), figures, paper macros
tests/                 simulator conservation, time reversal, estimators, readouts, model shapes
paper/                 ICML-style LaTeX, compiled with Tectonic
docs/                  literature check (all arXiv ids verified), submission checklist, Russian overview
```

## Related repos

Part of a from-scratch physical-AI series: [ARMageddon](https://github.com/nonberw/armageddon),
[YOINK](https://github.com/nonberw/yoink), [COPYCAT](https://github.com/nonberw/copycat),
[ZOOMIES](https://github.com/nonberw/zoomies), [LISTEN](https://github.com/nonberw/listen),
[SPUD](https://github.com/nonberw/spud).

## License

MIT.

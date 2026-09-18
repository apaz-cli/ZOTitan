# Overnight Report: autonomous MeZO hyperparameter exploration

Ran ~8h autonomously (timer PID logged, expired on schedule). Repo: `~/git/ZOTitan`.
Started from the state in `FINDINGS.md` and swept the knobs we hadn't touched.

## What I ran

1. **Perturbation distribution** — gaussian vs LOZO vs ZO-Muon(polar). Hit and fixed a
   bfloat16 `torch.linalg.qr` bug in the polar path (cast to float32).
2. **Momentum head-to-head** — settled RMSProp (momentum=none) vs ZO-Adam.
3. **Polar/LOZO rank** — rank 2..256.
4. **Fitness shaping** — MeZO slope vs OpenAI-ES (centered_rank) vs GRPO.
5. **shared_batch** on/off × z_batch × lr.
6. **Compute-normalized split** — fixed tokens/step, varying batch×z.
7. **Training horizon × lr** — 1k/2k/4k steps (this was the big one).
8. **Longer horizon** — 6k/8k steps (partial; timer expired).

## New findings (delta over FINDINGS.md)

- **Optimizer structure confirmed:** second moment is everything (~0.4), momentum is
  neutral-to-slightly-harmful. ZO-RMSProp (momentum=none + second moment) is the
  canonical optimizer; it beat ZO-Adam 2.386 vs 2.399.

- **Perturbation distribution barely matters.** gaussian / LOZO / ZO-Muon are within
  ~0.05 loss, and run-to-run noise is ~0.03, so the differences are mostly noise.
  ZO-Muon's apparent win (2.370) did not reproduce (2.401 on rerun). Not a big lever
  at this model size. (Polar rank 8–16 is the mild sweet spot if you use it.)

- **Fitness shaping (ES / GRPO) does not beat the plain ±ε slope.** none 2.328 <
  GRPO 2.346 < ES 2.377. The ranking estimators add nothing here.

- **shared_batch=True helps MeZO** by a consistent ~0.05–0.08 across all z_batch —
  contrary to the code's own docstring (which predicted a noise floor). Real and robust.

- **Compute frontier:** at fixed tokens/step the sweet spot is batch≈16–32 with
  z≈16–32; more total tokens per step keeps improving loss (still under-trained at
  1000 steps). Best at 1k steps: bs32_z32 → 2.182 (vs 2.394 original).

- **★ The headline: the training horizon is a confound on everything.**
  The optimal lr *decreases* with steps:
      1000 → 2.5e-3 (2.164),  2000 → 2e-3 (2.086),  4000 → 1.5e-3 (2.052),
      6000 → 1.5e-3 (2.039),  8000 → 1.5e-3 ≈ 2e-3 (2.056 / 2.062, tied).
  So the entire lr story from the day — "2.5e-3 is optimal, 4e-3 is a cliff, lr
  scales down with model size" — is an artifact of the 1000-step budget. At longer
  horizons the optimum slides down to ~1.5–2e-3 and plateaus there (1e-3 is then
  too low, 2.44).

## Best config now (this whole line of work)

- RMSProp (momentum=none, second_moment=on), shared_batch=True, bs32_z32,
  eps=1.2e-3, lr=1.5e-3, 6000 steps → **loss 2.039** (started at 2.394).

## Recommendations for next time

- Re-run the lr-vs-width/depth scaling at a *matched longer horizon* — the
  "lr ∝ 1/√N" law was measured at 1000 steps and is likely distorted.
- Tune β₂ / denom_eps (untouched; the second moment is the dominant ingredient, so
  its shape is the natural next lever).
- The model is still far from converged at 6k steps (loss 2.04 on a byte vocab);
  real conclusions about optima need 20k+ steps or a compute-matched budget.

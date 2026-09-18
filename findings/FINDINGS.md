# Findings: MeZO hyperparameter landscape (tinystories GRU)

Model: 2-layer byte GRU LM (`zotitan/tinystories-byte-rnn`), body ~10M params
(hidden 1024), MeZO zeroth-order training, 1000 steps, bs16_z16. Metric = min
train loss (lower is better). Divergence = loss > 10 (loss-kill).

## Bullet learnings

- **lr is the master knob.** Sharp optimum ~2.5e-3; below 1e-3 under-trains,
  above ~4e-3 diverges (loss climbs and explodes). ±20% costs ~0.08 loss.

- **eps is a wide flat plateau.** Good anywhere in ~[4e-4, 1e-2], optimum ~1.2e-3.
  Scale-free: the SPSA estimate (L+ − L−)/2ε cancels ε to first order. Curvature
  bias kills it only past ~3e-2. Independent of lr (no interaction).

- **More z-samples is better.** batch_split ordering: bs16_z16 ≈ bs8_z32 >
  bs32_z8 > bs64_z4. Gradient-estimate variance ~ 1/z_batch. bs64_z4 dominated.

- **lr scales down with model size.** Optimal lr: 25M→2e-3, 50M→1.5e-3,
  100M→1e-3 (width); 1 layer→2.5e-3, 5 layers→1.5e-3 (depth). Roughly lr ∝ 1/√N.

- **Width ≫ depth at fixed step budget.** Width scaling degrades loss gently
  (10M→2.39, 100M→2.61); depth scaling degrades hard (1L→2.21, 5L→3.05). Deep
  GRUs under-train severely in 1000 steps. Width triggers divergence at high lr;
  depth never diverges, just trains worse.

- **Clipping is useless.** proj-grad clipping (per_pair / norm, adaptive
  quantile) neither prevents divergence nor improves loss; it slightly hurts.
  Redundant with the second-moment normalization.

- **Optimizer structure: second moment is everything, momentum is dead weight.**
  +second moment ≈ −0.4 loss. Momentum is neutral-to-slightly-harmful (−0.03).
  The lr landscape (optimum 2.5e-3, cliff ~4e-3) is optimizer-invariant — same
  for plain MeZO, RMSProp, momentum, Adam. ZO-RMSProp ≥ ZO-Adam.

- **Core insight: ZO-Adam ≈ sign-MeZO.** The second-moment denominator divides
  the gradient magnitude out of the update, leaving a fixed-magnitude step ~lr.
  This single fact explains lr's sharpness (fixed step), eps's scale-freedom
  (g ∝ ε cancels), and clipping's irrelevance (g cancels).

## Open questions for overnight runs

- Perturbation distribution (gaussian / LOZO / ZO-Muon polar) — untested, big
  potential.
- β₂ / denom_eps shape of the second moment (the real lever on the lr cliff).
- Fitness shaping (ES centered-rank / GRPO) — different estimator entirely.
- 1000-step confound: does the lr optimum / size ordering change at 10k steps?
- z_batch × batch at fixed compute (efficiency frontier).

# Findings: MeZO hyperparameter landscape (tinystories GRU)

Model: 2-layer byte GRU LM (`zotitan/tinystories-byte-rnn`), body ~10M params
(hidden 1024), MeZO zeroth-order training, 1000 steps unless noted. Metric = min
train loss. Divergence = loss > 10 (loss-kill).

## Bullet learnings

- **lr is the master knob.** Sharp optimum ~2.5e-3 at 1000 steps; below 1e-3
  under-trains, above ~4e-3 diverges. Nothing else we swept moves the needle
  anywhere near as much.

- **eps is a wide flat plateau** (~[4e-4, 1e-2], optimum ~1e-3). The SPSA estimate
  (L+ − L−)/2ε cancels ε to first order, so eps is nearly scale-free; curvature
  bias only kills it past ~3e-2. Independent of lr.

- **ZO-Adam ≈ sign-SGD.** The second moment with denom_eps=1e-8 turns the update
  into `lr · sign(g)` — a fixed-magnitude step. Consequence: **the loss floor is set
  by lr** (a fixed step can't settle; it bounces at a floor). This is why lr is so
  sharp and why "start hot, then back off" (lr-guard) never beat the tuned lr.

- **Second moment is essential, momentum is dead weight.** +second_moment ≈ −0.4
  loss. Momentum is neutral-to-slightly-harmful (−0.03). Best optimizer:
  RMSProp (momentum=none + second_moment) ≈ ZO-Adam. The lr landscape (optimum,
  cliff) is optimizer-invariant.

- **The horizon is a confound.** Optimal lr *decreases* with steps: 2.5e-3 @1k →
  2e-3 @2k → 1.5e-3 @4k → plateaus ~1.5e-3 @6k. Every 1k-step lr "optimum" is an
  artifact of the short budget.

- **lr scales down with model size**, roughly ∝ 1/√N (width: 25M→2e-3, 100M→1e-3;
  depth similar). Measured at 1k steps, so also horizon-confounded.

- **More z-samples is better** (gradient variance ~1/z_batch): bs16_z16 ≈ bs8_z32 >
  bs32_z8 > bs64_z4. `shared_batch=True` helps MeZO ~0.06 (contrary to the code's
  own docstring).

- **Null results** (tested, don't help): gradient clipping (redundant with second
  moment), fitness shaping (ES/GRPO ≤ plain slope), perturbation distribution
  (gaussian ≈ LOZO ≈ ZO-Muon, within ~0.03 noise), contraction/feedback
  regularizers (flat-to-negative).

- **Self-correction ("counteraction", arXiv:2609.11716) is an emergent property,
  not a training lever.** The feedback term F_ℓ = ⟨Δh_{ℓ−1}, Δh_ℓ − Δh_{ℓ−1}⟩ is
  negative (block damps perturbations) even at random init for a GRU (saturating
  gates), and it's *isotropic* — it damps signal and noise directions equally.
  The signal/noise selectivity lives in the absolute flow (cos(h,u)>0), not the
  differential F. Four attempts to exploit it (contraction regularizer,
  stiff-subspace weighting, layerwise lr, head regularizer) were flat-to-negative.
  It's useful only as *telemetry*: fb_rel_2 (normalized hidden divergence) reads
  out the distance-to-the-stability-cliff and flags divergence ~100 steps before
  the loss turns.

- **LM-head geometry (the paper's 2nd mechanism) reproduces:** top-k token vectors
  align with the hidden state over training (cos_topk 0.07 → 0.14) and their logits
  stay more stable under weight perturbation than bottom-k's.

## Open levers (priorities for the scaling work)

- `denom_eps` (soft sign-SGD → lower floor) and `β₂` — untested optimizer-shape knobs.
- WSD lr schedule (`lr_wsd.warmup_frac` / `decay_frac` in code) — never turned on.
- The lr-vs-size law needs a *matched longer horizon* and a µP-style transfer rule.

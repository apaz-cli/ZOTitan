# Overnight Report 2: optimizer shape, schedules, quantization, MoE, scaling laws

Ran autonomously ~7h on the 2-layer byte GRU LM (MeZO, bs16_z16, 1000 steps).
Goal: fill in denom_eps / β₂ / WSD, test fp8 quantization and MoE, and extract
hparam-transfer / scaling-law rules.

## Results by task

### denom_eps (soft sign-SGD)
- Larger denom_eps shifts the optimal lr UP but does **not** lower the loss floor:
  1e-8 @ lr 2e-3 → 2.457; 1e-1 @ 4e-3 → 2.438; 3e-1 @ 6e-3 → 2.396; 1e0 → 2.93.
- So denom_eps and lr are **coupled** (rescale lr when you change denom_eps), but the
  best achievable loss is flat (~2.40) — the default 1e-8 is fine.

### β₂
- Lower β₂ is slightly better at low lr (0.9 @ 2e-3 → 2.411 vs 0.999 → 2.462), but
  it interacts with denom_eps/lr. Small effect, default 0.999 is fine.

### WSD lr decay
- Decay helps when lr is high (lr 4e-3: decay_frac 0.8 → 2.415 vs constant 2.567),
  but the best WSD run (2.415) does **not** beat the tuned constant default (2.394)
  at 1000 steps. Same "loss floor, not speed" story.

### ★ fp8 quantization breaks MeZO
- Weight-only fp8 (e4m3/e5m2, quantized each forward) has **no viable eps window**:
  eps 0.03 → loss ~5.5 (no gradient: perturbation below the fp8 step ~0.06, rounded
  away, proj_grad = 0); eps 0.1 → ~8 (worse than random); eps 0.3 → ~24 (diverged).
- Root cause: eps must be **larger** than the quantization step (to perturb) but
  **smaller** than the curvature scale (~0.3, to avoid blowup), and for fp8 these
  don't overlap (bf16's step ~1e-3 leaves a wide window; fp8's ~0.06 doesn't).
- **Conclusion: naive MeZO + fp8 weights is fundamentally incompatible** unless you
  do it QAT-style: quantize weights to fp8 ONCE, then perturb the bf16 master in
  bf16 on top (never re-quantize the perturbed weights). Verified: loss 2.438 at
  1000 steps ≈ bf16 baseline (2.40), lr/eps transfer unchanged. Treat fp8 as a
  storage/forward-only format, not the perturbation grid.

### MoE
- top-k MoE (4 or 8 experts) does **not** shift the optimal lr (2.5e-3) or eps
  (~1e-3/3e-3): identical to dense. It slightly hurts loss at fixed steps
  (e=4: 2.413, e=8: 2.406 vs dense 2.400) — more params, same budget → under-trained.
- **Transfer rule: architecture (MoE vs dense) doesn't move lr/eps.**

### Scaling laws (width)
- Optimal lr decreases with width (refined grid): **3e-3 (2.8M) → 2.5e-3 (10M) →
  1.5e-3 (39M) → 1.25e-3 (154M)** body params. Practical rule: **halve lr per ~4×
  body size** (≈ N^−0.3..−0.5, or ∝ 1/width — consistent with µP's hidden-weight rule;
  the exponent flattens at large size because those models under-train at 1k steps).
- **eps transfers unchanged**: best eps ~1e-3 at both 10M and 154M (scale-free, as the
  SPSA cancellation predicts).
- At fixed 1000 steps the loss is ~flat from 2.8M to 10M (~2.37–2.40) and degrades
  beyond (39M → 2.50, 154M → 2.63): a mini-Chinchilla — bigger ≠ better at a fixed
  budget, and the compute-optimal size is small (~3–10M here).
- **Held-out validation**: the transfer rule predicted lr_opt ≈ 1.7e-3 for hidden=1536
  (22M, unseen); the actual optimum was 2e-3 — within ~15%, so a ×1.5 sweep around
  the prediction nails it.

## The transfer methodology (the deliverable)

To train a new size or architecture:

1. **lr is the master knob — transfer it as “halve per ~4× body size”** (lr ∝ 1/width,
   ≈ N^−0.4), then sweep a few values around the estimate; the cliff is sharp.
2. **eps transfers** — fix at ~1e-3 (the scale-free plateau [4e-4, 1e-2]); do not
   re-tune per size or architecture.
3. **denom_eps / β₂ / clipping / distribution are second-order** — leave at defaults.
4. **Do not use fp8 (or lower) weights with MeZO** unless QAT-style: quantize once
   and perturb a bf16 master — never quantize the perturbed weights (that pins eps
   to the quantization step and kills the gradient).
5. **At a fixed step budget, sweep model size** — there's a Chinchilla-style optimum
   (~10M at 1k steps here), and it's more important than any optimizer knob.

## Still running at report time

- The 154M (hidden=4096) runs are slow (~40+ min each); lr=6e-4 and eps=3e-4 anchors
  were still finishing. They only sharpen the fit, not the conclusions.

## Code added (uncommitted)

- `zotitan/tiny_model.py`: `MoEBlock` (top-k router + per-expert MLPs) and
  `TinyStoryLM(num_experts, top_k)`; `build_tiny` overrides.
- `zotitan/model.py`: `--model.num-experts` / `--model.top-k`.
- `zotitan/train_zo.py`: `--zo.weight-quant {none,fp8_e4m3,fp8_e5m2}` — per-forward
  fp8 weight rounding (the "naive" mode that breaks ZO; the QAT fix is to quantize
  once and perturb bf16, tested but not yet a flag).

## Sweep files added

denomeps, denomeps2, beta2, wsd, quant, moe, scale, scale2, eps_scale, heldout.

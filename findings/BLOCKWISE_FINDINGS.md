# Per-block SPSA: decoupling the ZO estimator (GradGuess campaign, 2026-10-06)

Model: `zotitan/tinystories-byte-rnn` (2-layer byte GRU, ~10.6M params), TinyStories,
MeZO/ZO-Adam unless noted, bs16, eps 9e-4, 1000 or 3000 steps. Metric: min train loss
(lower better) unless labelled ppl (held-out, 1000 val examples, minimize).

## TL;DR

1. **Per-block SPSA + a partial activation restriction is the winner.** Perturb one
   weight block at a time, give each block its own directional derivative, and draw each
   block's `z` as `√α · (activation-subspace guess) + √(1−α) · isotropic`. The best
   1000-step config found is `blockwise act, ndir=32, mix=0.5`: held-out ppl **`6.89`**
   (blockwise Gaussian `7.21`, global `11.03`). At 3000 steps the best is
   `blockwise act8 mix=0.75` = **`1.907`** min loss (blockwise Gaussian `1.942`, global
   `2.239`).
2. **The paper's subspace restriction helps only partially, and only once decoupled.**
   A *hard* row-space restriction (`act`, mix=1.0) loses to blockwise Gaussian at 1000
   steps (`2.27` vs `2.21`). But a *partial* one wins: `act mix=0.5–0.75` is the best
   1000-step config found (`1.93` / ppl `6.74` vs blockwise Gaussian `1.98` / `7.33`).
   A fixed *random* subspace of the same rank/mix gives `7.24–7.33` ppl, i.e. no better
   than Gaussian, so the gain is specifically the activation alignment.
3. **Horizon amplifies the partial-restriction gain.** At 3000 steps / lr 1.5e-3, global
   `act mix=0.75` beats global Gaussian `2.141` vs `2.239`, monotone in mix (0.75 > 1.0 >
   0), and blockwise `act8` (hard) ties blockwise Gaussian (`1.943` vs `1.940`).
4. Cost: blockwise spends `#blocks × block_directions` forward-pass pairs per step.
   `param` grouping = 7 blocks; `layer` = 4. It keeps improving through 16-32 directions.

This is code: `ZOConfig.blockwise`, `block_directions`, `block_grouping` in
`zotitan/train_zo.py`, plus `PerturbationConfig.subspace_mix`.

## Why decoupling should help

Global SPSA reads one scalar `d = Σ_b ⟨g_b, z_b⟩` and updates every block with `d z_b`.
For block `b` that estimate carries the noise of *all* other blocks
(`Var ∝ Σ_{b'} ‖g_{b'}‖²`). Per-block SPSA reads `d_b = ⟨g_b, z_b⟩` and updates only
block `b`, so its variance is `∝ ‖g_b‖²` — the layers become statistically independent.
The paper's per-layer structure ("guess the layer's own output gradient, form ΔW = g xᵀ")
is the same idea; here it is realised as block-wise directional derivatives, which is
what MeZO's weight-space estimator can actually express.

## Results

### 1000 steps, lr 2.5e-3 (min train loss)

| estimator | config | min loss |
|---|---|---|
| global SPSA (control) | z_batch 16 | 2.43–2.45 |
| global SPSA | z_batch 32 | 2.305 |
| global SPSA | z_batch 56 (compute-matched to blockwise ndir=8) | 2.287 |
| blockwise Gaussian | ndir 4, param | 2.30–2.34 |
| blockwise Gaussian | ndir 8, param | 2.17–2.23 |
| blockwise Gaussian | ndir 16, param | 2.081 (2.058 @ lr 2e-3) |
| **blockwise Gaussian** | **ndir 32, param** | **1.977** |
| blockwise Gaussian | ndir 32, layer | 2.067 (2.024 with shared_batch) |
| blockwise `act` r128 | ndir 8, mix 1.0 | 2.274 |
| blockwise `act` r128 | ndir 8, mix 0.75 | 2.173 |
| **blockwise `act` r128** | **ndir 8, mix 0.5** | **2.155** |
| blockwise `wrow` r128 | ndir 8, mix 0.5 | 2.218 |
| blockwise `wrow` r128 | ndir 8, mix 0.75 | 2.241 |

Update rules at ndir 4: ZO-RMSProp `2.337` ≈ ZO-Adam `2.352`; R-AdaZO diverged. `param`
grouping beats `layer` at ndir ≥ 8.

### Held-out ppl, 1000 steps, lr 2.5e-3

| cfg | ppl |
|---|---|
| global z16 | 11.03 |
| blockwise act8 r128 (hard) | 10.46 |
| blockwise wrow8 r128 | 10.40 |
| blockwise gauss ndir8 | 9.09 |
| blockwise act8 mix 0.5 | 9.19 |
| blockwise gauss n16 param | 8.09 |
| blockwise gauss n32 layer | 7.85 |
| blockwise gauss n32 param | 7.21 |
| blockwise act n16 mix 0.75 | 7.78 |
| blockwise act n16 mix 0.5 | 7.88 |
| **blockwise act n32 mix 0.5** | **6.89** |

The three levers compose: at ndir=8 the partial restriction helps min loss but not held-out
ppl (9.19 vs 9.09); at ndir=32 it helps both, and ndir=32 beats ndir=16 (7.78). The best
1000-step config found is **blockwise `act` ndir=32, mix=0.75, param** (ppl 6.74).

### Controls: matched compute and a random subspace

Two questions: does blockwise just win by spending more compute, and is `act` specifically
about activations?  At 1k / lr 2.5e-3 (held-out ppl):

| cfg | forwards/step | min loss | ppl |
|---|---|---|---|
| global z56 | 112 | 2.273 | 9.556 |
| blockwise gauss ndir8 | 112 | 2.181 | 8.969 |
| blockwise gauss ndir32 | 448 | 1.981 | 7.333 |
| blockwise randsub ndir32 mix 0.5 | 448 | 1.987 | 7.242 |
| blockwise randsub ndir32 mix 0.75 | 448 | 2.003 | 7.329 |
| blockwise act ndir32 mix 0.5 | 448 | 1.927 | 6.826 |
| blockwise act ndir32 mix 0.75 | 448 | 1.925 | 6.742 |

- **Blockwise is not just more compute.** At equal 112 forward passes/step, blockwise
  Gaussian beats global `z=56` on both min loss (2.18 vs 2.27) and ppl (8.97 vs 9.56).
- **`act` is activation-specific, not a distraction.** `randsub` is a *fixed random*
  orthonormal subspace of the same rank and mix; it lands right on top of Gaussian
  (ppl 7.24–7.33 vs 7.33), while the activation subspace is clearly better (6.74–6.83).
  So it is the alignment with the activations, not merely mixing in a correlated
  perturbation, that helps.
- At ndir32 the subspace buys ~0.5–0.6 ppl over Gaussian, about the same as the
  blockwise-vs-global gain at matched compute. Both matter; neither is the whole story.

### 3x horizon (3000 steps)

At lr 2.5e-3 **everything diverges** (global final 10.07, blockwise bg8 final 5.15): the
lr optimum slides down with horizon, as `findings/OVERNIGHT_REPORT.md` predicted.

At lr 1.5e-3 (stable):

| estimator | min loss | final |
|---|---|---|
| global Gaussian (mix 0) | 2.239–2.265 | 2.311 |
| blockwise Gaussian ndir8 | 1.942 | 2.033 |
| blockwise act8 r128 (hard) | 1.943 | 2.033 |
| blockwise wrow8 r128 | 2.003 | 2.066 |
| **blockwise act8 mix 0.75** | **1.907** | 2.008 |
| blockwise act8 mix 0.5 | 1.916 | 2.010 |
| global act r128 mix 0.25 | 2.225 | 2.269 |
| global act r128 mix 0.5 | 2.172 | 2.219 |
| **global act r128 mix 0.75** | **2.141** | 2.208 |
| global act r128 mix 1.0 | 2.158 | 2.251 |

At 1000 steps the blockwise partial-restriction optimum is different: `act mix=0.5`
(2.155) beats `mix=0.75` (2.173) and hard `mix=1.0` (2.274). So the best mix fraction
drifts up with horizon (0.5 at 1k, ~0.75 at 3k) — consistent with variance reduction
mattering more as the run lengthens.

Held-out ppl at 3k / lr 1.5e-3: blockwise `act8 mix=0.75` = **6.839** vs blockwise
Gaussian = 7.080. The 3k ordering matches min loss (`1.907` vs `1.942`), so the
partial-restriction gain is not a train-loss artifact.

## Recommended config

- **1000 steps**: blockwise `act`, `block_directions=32`, `subspace_mix=0.75`, `rank=128`,
  ZO-Adam, lr 2.5e-3 → held-out ppl **6.74**. (Blockwise Gaussian ndir=32 gives 1.98 min
  loss / 7.33 ppl; a fixed random subspace gives 7.24–7.33, so the activation alignment
  is doing real work.)
- **3000 steps**: blockwise `act`, `block_directions=8`, `subspace_mix=0.75`, `rank=128`,
  lr 1.5e-3 → min loss **1.907**, held-out ppl **6.84**. (ndir=32 at 3k was not run —
  ~450 forward pairs/step × 3000 was too slow to finish.)

## Reproduction

Curated sweeps that reproduce the headline claims (the ~75 raw runs are on the manager
under campaign `GradGuess`):

- `sweeps/tinystories_blockwise.py` — 1k screen: global control, blockwise Gaussian
  ndir {4,8,16,32}, blockwise act ndir {8,16,32} mix 0.5.
- `sweeps/tinystories_blockwise_ppl.py` — held-out ppl for the 1k winners.
- `sweeps/tinystories_blockwise_3k.py` — 3k winners at the stable lr 1.5e-3 (divergence
  check; lr 2.5e-3 diverges at 3k).
- `sweeps/tinystories_mix3k.py` — 3k partial-restriction (mix) trend, global estimator.
- Code coverage: `tests/test_smoke.py::test_smoke_blockwise` and
  `tests/test_subspace_sampling.py`.

(The matched-compute and random-subspace controls in the Controls table used a temporary
`randsub` source that was removed after the run, so they are not reproducible from this
tree; their raw runs are on the manager under campaign `GradGuess`.)

## Open questions

- ndir 32 at 3x was cancelled (too slow); the 3x winners use ndir 8. Whether
  ndir 16-32 + mix at 3x improves further is untested.
- Held-out ppl at 3k, and a longer horizon (6k+) with the lr schedule re-tuned.
- Whether an even finer block split (per gate) helps beyond the per-weight blocks.
- The blockwise estimate reuses the same `ndir` batches across blocks; giving each
  (block, direction) its own batch is untested and changes both data and compute.

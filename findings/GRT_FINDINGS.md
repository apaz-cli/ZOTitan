# Can zeroth-order training make a looped transformer work at depth?

Status as of 2026-09-26 14:00. Model: the Gated Recurrent Transformer (GRT,
arXiv 2608.15062), medium isoFLOP checkpoint (2 prelude blocks, 5 shared blocks looped R
times, 2 coda blocks; trained at R = 4). Code is in `zotitan/grt.py`, sweeps in
`sweeps/grt/`, results in `~/.mlsweep/experiments/<id>/`.

## Summary

1. The published checkpoint collapses past its trained depth. Val loss goes from 2.84 at
   R = 4 to 4.87 at R = 64 and 7.42 at R = 512. LAMBADA accuracy falls from 34% to 2%.
2. Backprop through the loop stops working at depth. At R = 256 the exact gradient is 10⁵×
   too large and uncorrelated with the actual change in loss. The cause is chaotic dynamics:
   fp32 backprop fails the same way bf16 does. Finite differences, which is what ZO measures,
   stay meaningful.
3. A little shallow FO training removes that chaos. After 750 FO steps at R ≤ 8, backprop is
   reliable even at R = 1024. For this model FO has a curriculum escape, so the chaotic-gradient
   barrier is real but easy to get around.
4. The fix lives in the gate. FO on the 3.15M-parameter gate MLP alone repairs deep loss about
   as well as FO on all 68M recurrence parameters.
5. ZO works once it trains the right parameters at the right depths. Training just the gate
   and W_proj (5.25M params) with R sampled between 4 and 64 cut R = 64 loss from 4.87 to 3.82 in 250
   steps while R = 4 moved by 0.05. It did this by raising the gate at depth, which makes the
   loop more contractive. The earlier ZO setups (all 68M params, or R sampled from 1) barely
   moved.
6. At depth, FO wins decisively on this model. Backprop through up to 256 loops (R sampled
   log-uniformly from 4 to 256) flattened the whole curve within 200 steps: 2.94 at R = 512,
   against 6.51 for ZO after 250 steps. FO's gradient spiked to a norm of 97 in its first
   steps, clipping absorbed it, and the loop became contractive right away. The chaotic-gradient
   barrier is real in the untrained model but did not stop FO in practice.

## Setup notes

The port is exact: fp32 logits match the reference `model.py` bit for bit.

The checkpoints were trained on OpenWebText. The README says cccc_filtered, but on that corpus
the models score 0.3 to 0.4 nats worse than reported. I rebuilt nanoGPT's exact OWT val split
(4,434,606 tokens; nanoGPT documents 4,434,897) and evaluated the full split at the trained R
with noise off:

| checkpoint | reported loss | ours | difference |
|---|---|---|---|
| medium isoFLOP | 2.896 | 2.901 | +0.005 |
| medium isoParam | 2.763 | 2.753 | −0.010 |
| large isoFLOP | 2.777 | 2.774 | −0.003 |
| large isoParam | 2.636 | 2.639 | +0.003 |

The released `estimate_loss` (noise on, random R) reads 0.1 to 0.19 higher than the reported
numbers, so those came from a deterministic eval. Everything below uses OWT (the `owt`
objective).

Other things to know:

- The two medium HF repos hold each other's weights. `build_grt` picks the repo by layout. The
  trained R (4 for medium, 6 for large) is missing from `config.json`.
- ZO needs common random numbers on this model. GRT draws noise, and the objective draws R, on
  every forward pass. Without replaying the RNG for the + and − passes, a pair's loss
  difference reached 0.9 at z = 0. `ZOConfig.crn` (on by default) makes it exactly 0.
- Precision and ε, checked against backprop at R = 4: fp16 forward, fp32 master weights,
  ε = 1e-3. ε = 1e-2 is too coarse for this model, and ε = 1e-4 fails under bf16.
- Periodic evals use the same 64 val windows in every run, so runs compare exactly. Values from
  64 windows sit about 0.06 below full-split numbers.

## 1. The baseline collapses past its trained depth

Val loss (64 windows) and LAMBADA accuracy (5,153 examples) by loop count:

| R | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|---|---|---|
| val loss | 3.83 | 3.17 | **2.84** | 2.96 | 3.31 | 3.95 | 4.87 | 5.85 | 6.71 | 7.42 |
| LAMBADA % | 12.8 | 25.3 | **34.0** | 25.8 | 14.4 | 5.5 | 1.7 | | | |

The large checkpoint does the same around its R = 6. Its LAMBADA score at R = 6 is 39.1%,
matching the paper's 39.05%, and it drops to 6.0% at R = 64. The fixed-point residual
‖hᵣ − hᵣ₋₁‖/‖hᵣ₋₁‖ settles near 0.01 and stays there, so the state drifts instead of
converging.

Freezing each token once its relative update drops below τ = 0.2 turns the collapse into a
plateau: loss 2.86 at about 5 loops, flat out to R = 512 (LAMBADA 32.3%). It never beats
fixed R = 4.

After training at R ≤ 8 (section 3), accuracy holds up at depth, but no checkpoint beats its
best shallow depth (about R = 3 or 4):

| checkpoint | R4 | R16 | R64 | R128 |
|---|---|---|---|---|
| base | 34.0 | 14.4 | 1.7 | |
| FO, all 68M loop params, lr 3e-5 | 34.8 | 32.9 | 26.3 | 19.7 |
| FO on gate + W_proj | 34.3 | 32.6 | 28.0 | 23.7 |
| ZO, all 68M loop params, 600 steps | 35.3 | 18.1 | 2.7 | 0.7 |

Extra loops stop hurting, but they do not help. LAMBADA on the deep-trained checkpoints is
queued (`p5_lambada_deep.py`).


## 2. Why FO could fail at depth, and why it doesn't here

For a fixed batch with pinned noise, I compared the backprop gradient (what FO uses) against
fp32 finite differences along 32 random directions (what ZO uses). A correlation of 1 and a
slope of 1 mean backprop agrees with the real change in loss.

| checkpoint | R | ‖g‖² | correlation | slope |
|---|---|---|---|---|
| base | 4 | 12.5 | 0.998 | 0.99 |
| base | 16 | 60 to 68 | 0.99 | 1.0 |
| base | 64 | 1.3e3 to 1.8e3 | 0.90 | 0.72 |
| base | 256 | 1e6 to 8e6 | 0.16 to 0.21 | 0.01 |
| base | 1024 | 5e10 to 9e14 | about 0 | 0.000 |
| after FO at R ≤ 8 | 256 | 114 to 134 | 0.82 to 0.89 | 0.99 |
| after FO at R ≤ 8 | 1024 | 477 to 505 | 0.73 to 0.88 | 1.14 |

Switching backprop to fp32 changes nothing, which rules out rounding. At R = 256 the
finite-difference slope also depends on ε: correlation is 0.5 to 0.6 at ε = 1e-4, 0.2 at
1e-3, and 0 at 1e-2. The loss surface at depth has a large, short-wavelength component that
dominates the exact gradient, and a finite ε averages it out. This matches the regime in Metz
et al. 2021 ("Gradients are Not All You Need"), where the gradient of a smoothed objective,
which ES and ZO estimate, is the useful signal.

The bottom two rows are the catch. After a little training at shallow depth, the loop becomes
contractive and backprop is accurate out to R = 1024. The chaos comes from a model that never
saw depth. Looped transformers are not chaotic by nature.

So on this model, ZO's advantage has to come from somewhere other than FO being unable to
start. Candidates: objectives FO cannot differentiate (discrete halting), memory (checkpointed
FO stores one state per loop step and ZO stores none), or regimes where the loop has to stay
rich instead of contractive.

## 3. Training results

### Shallow training, R from 1 to 8

At R ≤ 8, FO backprops through at most 44 layer applications, which is ordinary depth. This
phase mostly calibrated lr. FO on the 68M recurrence parameters for 750 steps:

| FO lr | R4 | R8 | R16 | R32 | R64 |
|---|---|---|---|---|---|
| base | 2.841 | 2.962 | 3.307 | 3.946 | 4.871 |
| 3e-6 | 2.857 | 2.886 | 2.994 | 3.218 | 3.628 |
| 1e-5 | 2.861 | 2.875 | 2.940 | 3.075 | 3.328 |
| 3e-5 | 2.871 | 2.878 | 2.918 | 2.998 | 3.147 |
| 1e-4 | 2.910 | 2.916 | 2.940 | 2.986 | 3.067 |

Training only at R ≤ 8 fixes depths far past 8. ZO in the same setup (Z = 64, lr 3e-5,
600 steps) moved the same way, slowly: R = 64 went from 4.87 to 4.74.

ZO was slow because its steps were almost all random walk. At lr 3e-5 ZO moved the weights as
far as FO did (‖ΔW‖² of 47.7 vs 57.2), but the RMS change was 8.4e-4 in every parameter group,
which is exactly lr·√steps. ZO-RMSProp moves each coordinate about ±lr per step regardless of
signal. FO at lr 3e-6 moved the weights 16× less and gained 10× more at R = 64.

FO spent 13 to 24% of its weight change on the gate, which holds 4.6% of the recurrence
parameters. Training subsets on their own (FO, 750 steps):

| trained params | count | R4 | R16 | R32 | R64 |
|---|---|---|---|---|---|
| all recurrence, lr 3e-5 | 68.2M | 2.871 | 2.918 | 2.998 | 3.147 |
| gate only, lr 3e-4 | 3.15M | 2.867 | 2.915 | 2.991 | 3.131 |
| gate + W_proj, lr 3e-4 | 5.25M | 2.884 | 2.918 | 2.975 | 3.081 |
| W_proj only, lr 3e-4 | 2.10M | 2.884 | 3.018 | 3.238 | 3.637 |

ZO noise grows with the number of trained parameters, so this shrinks ZO's problem 13×.

One more trap: sampling shallow depths starves ZO. ZO on gate + W_proj with R from 1 to 8
improved only R = 1 (3.83 to 3.77) and left R ≥ 4 flat or slightly worse. Shallow exits have
much higher loss, so they dominate the single scalar each ZO direction measures. FO sees the
full gradient and fixes both. The deep runs therefore sample R from 4 upward.

### Deep ZO, R log-uniform from 4 to 64

Trainable: gate and W_proj (5.25M params). 64 directions × 8 sequences, T = 256, about 22 loops per
forward pass on average, about 7 s per step on the 5090:

| | R4 | R8 | R16 | R32 | R64 | R128 | R256 | R512 |
|---|---|---|---|---|---|---|---|---|
| base | 2.841 | 2.962 | 3.307 | 3.946 | 4.871 | 5.845 | 6.712 | 7.424 |
| lr 1e-4, 250 steps | 2.849 | 2.913 | 3.146 | 3.584 | 4.262 | 5.084 | 5.889 | 6.634 |
| lr 3e-4, 250 steps | 2.887 | 2.896 | 3.038 | 3.330 | 3.816 | 4.493 | 5.269 | 6.041 |

Progress was steady at every eval and still roughly linear at step 250. Depths past the
training range improved as well: R = 512 dropped 1.4 nats at lr 3e-4.

The mechanism matches what FO found. ZO raised the mean gate (more copying) at R = 64 from 0.86
to 0.92, and the relative residual at R = 512 fell from 0.0091 to 0.0026. The loop is about
3.5× more contractive.

### Deep FO, same sampler

FO with the same sampler (R log-uniform from 4 to K), T = 256, 64 sequences per step as 8
micro-batches, backprop through every loop with activation checkpointing. FO learning rates are 3e-4 for
the small subset and 3e-5 for the whole loop (the best K = 8 settings); ZO trains the gate
and W_proj:

| | steps | R4 | R64 | R256 | R512 |
|---|---|---|---|---|---|
| base | 0 | 2.841 | 4.865 | 6.664 | 7.198 |
| FO on gate + W_proj, K = 256 | 200 | 2.879 | 2.893 | 2.926 | 2.943 |
| FO on the whole loop, K = 256 | 500 | 2.867 | 2.882 | 2.926 | 2.949 |
| FO on the whole loop, K = 64 | 500 | 2.864 | 2.904 | 3.005 | 3.071 |
| ZO, K = 256, lr 1e-4 | 250 | 2.845 | 4.358 | 5.860 | 6.512 |
| ZO, K = 64, lr 3e-4 | 250 | 2.887 | 3.816 | 5.269 | 6.041 |

FO makes the loop nearly depth-invariant out to R = 512 at a cost of 0.03 nats at R = 4, and it
gets there in a couple hundred steps. Its gradient did spike early (max norm 97 in the first 20
steps at K = 256, median 0.9 afterwards, nothing non-finite), but clipping at 1.0 absorbed the
spikes and the first few updates made the dynamics contractive, after which backprop behaves.
This is the curriculum escape from section 2 happening on its own: the sampler's shallow draws
(R near 4) are stable, and they fix the deep behavior before the deep draws can do damage.

So for GRT, ZO is not needed to train at depth, and at matched steps it is far behind. ZO's
remaining case here is the non-differentiable one (hard halting, section 4), or a model whose
useful regime is the chaotic one that FO's gradient cannot follow.

## 4. Adaptive depth (halting) on the base model

A learned per-token halting head (1,025 params, initialized to the τ = 0.2 residual rule),
trained with ZO on CE plus λ × loops used. The hard stop decision has no gradient, so this is a
case ZO handles directly and FO would need a relaxation for. First finished run (λ = 0,
300 steps):

| rule | loss | mean loops |
|---|---|---|
| fixed R = 4 | **2.841** | 4.0 |
| residual rule, τ = 0.2 | 2.860 | 5.0 |
| learned head | 2.868 | 4.6 |

On the base model nothing beats fixed R = 4, because every loop past 4 hurts almost every
token. Halting gets interesting on a model where extra depth helps some tokens, which the deep
training produces. Halting on a deep-trained checkpoint is next.

## 5. What is running, and what went wrong overnight

Both GPU jobs died without a traceback at 02:16 and 02:19, and most of `/tmp/mlsweep` was
cleaned around 02:40. The manager kept counting the dead runs as in flight, so the 13 queued
jobs never started, and about 8 GPU-hours were lost. I couldn't read the kernel log to check
for an OOM kill. The ZO K = 64, lr 3e-4 row above comes from that run's salvaged step-250 eval.

mlsweep bugs worth fixing (I did not change mlsweep):

1. After a worker reconnects, the manager reassigns its in-flight runs to "the first N free
   GPUs" (`_manager_workers.py`, around line 615), since the resume message doesn't carry the
   real `gpu_ids`. A 4070 job got booked on the 5090, and about 14 runs were then sent to the
   busy 4070 and died of OOM while the 5090 sat idle.
2. After that, the manager ran two jobs per GPU despite `--jobs 1`, which suggests a duplicate
   worker connection.
3. Runs whose process has died stay "in flight" indefinitely and block the queue.

Running now: ZO deep at K = 256 (lr 1e-4, on the 5090) and FO deep on gate + W_proj at K = 64
(on the 4070). Queued: the remaining FO deep runs, the other halting runs, and LAMBADA on the
trained checkpoints.

## Reproducing

| step | file |
|---|---|
| baseline depth curves | `sweeps/grt/p1_depth_baseline.py` |
| ZO vs backprop fidelity | `sweeps/grt/p2_zo_fidelity.py`, `zotitan/diag_zo.py` |
| OWT check | `sweeps/grt/diag_owt_full.py`, `zotitan/diag_datasets.py` |
| backprop vs depth | `sweeps/grt/p3b_backprop_vs_depth.py`, `p3b_backprop_cause.py`, `p3b_backprop_trained.py` |
| shallow training | `sweeps/grt/p3a_fo_k8.py`, `p3a_zo_k8.py`, `p3a_fo_subsets.py` |
| deep training | `sweeps/grt/p3b_zo_deep.py`, `p3b_fo_deep.py` |
| halting | `sweeps/grt/p4a_owt_halt_head_base.py` |
| LAMBADA | `sweeps/grt/p5_lambada_base.py`, `p5_lambada_trained.py`, `zotitan/diag_lambada.py` |
| tables | `python sweeps/grt/trajectories.py <experiment_id>`, `sweeps/grt/summarize.py` |

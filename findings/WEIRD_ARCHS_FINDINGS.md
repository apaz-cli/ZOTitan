# Findings: weird (non-backprop) architectures under ZO (tinystories, byte level)

Overnight 2026-10-13. Question: with `sweeps/tinystories_baseline.py`'s training config held
fixed (ZO-Adam: stored-EMA momentum + second moment, lr 2.5e-3, eps 9e-4, bs16 x z16
independent batches, 1000 steps, byte tokens, seq 256), and only `--model.model-id` changed,
which architectures train well, especially ones backprop can't train?

Campaigns: `WeirdArchs` (search), `ArchRegistry` (validation of the registry refactor).
Sweeps: `sweeps/weird_archs{,2,3,4,5}.py`, `sweeps/weird_confirm.py`, `sweeps/es_lr_gru.py`,
`sweeps/es_archs.py`, `sweeps/es_confirm.py`. Archs: `zotitan/archs/`.

Metrics: rounds 1–5 rank by last-50-step mean train loss (one seed each). The confirmation
runs use val nll over 1000 held-out stories, 3 seeds each. Model init is unseeded, so
**run-to-run noise is ~0.02–0.03**; differences smaller than that aren't findings.

## Bullet learnings

- **Size dominates under ZO at this budget.** A 1-layer GRU (3.9M body) gets val 2.214, while
  the 2-layer baseline GRU (10.2M) gets 2.396. Every comparison has to be size-matched, and the
  depth sweep shows the same thing.

- **Best arch: a hashed n-gram MLP (`(ngram,max_n=3)`, 2.0M body): val 2.091 ± 0.014.** That's
  0.30 better than the baseline GRU and 0.12 better than the size-matched 1-layer GRU, with no
  recurrence, and it runs at 0.04 s/step against 0.14 s/step. It holds that level from 2.0M
  down to 0.9M body. Fewer orders is better:
  n3 < n4 < n6 < n8, i.e. every extra table costs more estimator noise than its context
  gains. It is *not* backprop-proof: hashed lookups are ordinary embeddings.

- **Quantized-state GRU trains as well as the GRU.** The state is rounded every step, so
  there's no BPTT. At ≥64 levels it ties the 1-layer GRU (256 levels: 2.218 ± 0.011 vs 2.214).
  16 levels costs ~0.35, and 8 or 4 levels diverge. Under ES, 256 levels is slightly *better*
  than the GRU (2.160 ± 0.004 vs 2.178 ± 0.016).

- **Hard decisions cost little under ZO-Adam, but more under ES.** Transformers with argmax
  attention (`hardattn`) or Heaviside FFN units (`stepffn`) land within ~0.07 of the soft
  transformer under ZO-Adam (all ~3.0). Under ES they fall 0.33–0.45 behind (3.04/3.16 vs 2.71).

- **Transformers are bad at this config** (~3.0 at 3M or 10M, ~2.7 under ES). The whole
  xformer family is backbone-limited here, so swapping parts can't be judged against the
  GRU/ngram numbers.

- **Things that failed:**
  - Binary step-function GRU gates diverge (loss → 6.8).
  - Ternary weights: 3.73. LUT network: 4.26.
  - Learned class-grams: unstable.
  - Latent-code ROSA and code-match attention diverge or stall.
  - VQ bottleneck: ~= its transformer (3.01).

- **Things that didn't help:**
  - ROSA suffix-match retrieval. After fixing an init-scale bug it's 2.155 vs 2.091 without it.
  - Word-grams, with fixed or learned (hard-threshold) segmentation: ~2.15–2.20, worse than
    plain n3.
  - Counters (2.34 vs gru1 2.26).
  - GRU + n-gram features (= gru1).
  - Self-conditioning: helps the 2-layer GRU (2.38 vs 2.43), but not at matched size.

- **ES vs ZO-Adam:** ES lr sweep on the GRU: 2.5e-3 is best for both gaussian ES and EGGROLL
  (low-rank 64), and the two are equivalent (val 2.455 vs 2.463). With the lr fixed, ES ≈ ZO-Adam
  on the leaders (ngram n3: 2.095 vs 2.091). ES is better for the 1-layer GRUs and worse for
  the 2-layer GRU.

## Confirmation (3 seeds, val nll)

| arch (`--model.model-id`) | body | ZO-Adam | ES |
|---|---|---|---|
| `(ngram,max_n=3)` | 2.0M | **2.091 ± 0.014** | **2.095 ± 0.005** |
| `(ngram,max_n=3,hidden=512)` | 1.5M | 2.092 ± 0.026 | |
| `(ngram,max_n=4)` | 2.6M | 2.135 ± 0.011 | 2.165 ± 0.005 |
| `wordgram` | 3.9M | 2.154 ± 0.003 | |
| `(ngram_rosa,max_n=3)` | 2.1M | 2.155 ± 0.008 | 2.175 ± 0.012 |
| `(gru,num_layers=1)` | 3.9M | 2.214 ± 0.007 | 2.178 ± 0.016 |
| `(quant_gru,levels=256,num_layers=1)` | 3.9M | 2.218 ± 0.011 | 2.160 ± 0.004 |
| `(quant_gru,levels=64,num_layers=1)` | 3.9M | 2.242 ± 0.001 | |
| `gru` (baseline) | 10.2M | 2.396 ± 0.016 | 2.477 ± 0.004 |

## Search rounds (1 seed, last-50 train loss)

Round 1 (`weird_archs.py`):
- ngram 2.308; selfcond_gru 2.380; gru 2.434; ngram_gru 2.441
- rosa_gru_copy 2.520; rosa_gru 2.549
- latch_deep 2.808; hardroute 2.811
- transformer 3M 3.002 / 10M 3.001; latch 3.034; halting 3.039; stoch_gru 3.043; stepffn 3.049;
  looped 3.057; hardattn 3.072; stochffn 3.125
- rosa_latent 3.80 (diverging); codemem 3.82; quant_gru-16 (2-layer) diverged; ternary_gru flat
  at 5.557 (no signal)

Round 2 (`weird_archs2.py`):
- ngram n4 2.191; ngram_gru1 2.249; gru1 2.258; quant64 2.258; ngram 16k-buckets 2.282
- rosa_gru1_copy 2.299; ngram h2048 2.305; ngram 2.310; counter 2.342; ngram small 2.350
- selfcond 2.36; gru1 h512 2.384; ngram n8 2.402
- latch 2.68–2.87; ngram_rosa 2.707 (init bug); hardroute-small 2.866
- vqformer 3.011; classgram 3.377; ternaryw 3.729; lutnet 4.256

Round 3 (`weird_archs3.py`):
- ngram n3 2.145; n3 d256 2.152; n4 h512 2.167; n4 d64 2.172; n2 2.174
- n4 variants 2.18–2.22; quant256 2.244; n5 2.250; gru1 2.260
- quant16 2.595; quant8 3.58 / quant4 4.49 (diverging); hard-gate GRU 6.6–6.8 (diverged)

Round 4 (`weird_archs4.py`):
- n3 variants 2.135–2.165
- wordgram 2.18–2.19; learned-boundary wordgram 2.185 / 2.203; wordgram n4 2.209
- quant256 2.258; gru1 2.262

Round 5 (`weird_archs5.py`, shrinking the leader):
- n3 d64 h512 2.132 (0.9M body); n3 d32 2.147 (0.9M); n3 h256 2.151; n3 2k-buckets 2.156
- n3 2.162; n2 h512 2.164
- n3 d_out256 2.288

The n3 family plateaus at ~2.13–2.16 from 2.0M down to 0.9M body. Shrinking the head input
(d_out 512 → 256) is the one cut that hurts.

## Validation of the arch-registry refactor (`ArchRegistry`)

- **Baseline A/B, 3 seeds each:** old code 2.420 ± 0.020 vs new code 2.400 ± 0.020 (min train
  loss). No regression.
- **Depth sweep** (5 depths x lr {1.5e-3, 2.5e-3}): every cell is within ±0.03 of its
  2026-09-17 value. The body sizes match the sweep docstring exactly.
- **Width sweep:** trimmed to 25m/50m x the same two lrs. The full grid is ~9 GPU-h.

## Bugs found along the way

- `losses._compiled_torch_xentropy` had its compile flag inverted (fixed).
- Local archs computed CE in bf16, which zeroes the ZO slope on the non-fused path (fixed).
- ROSA retrieval embeddings used std-1 init next to std-0.02 n-gram tables (fixed, round 5 on).
- Model init is unseeded, so runs aren't reproducible. Not changed: it would alter every sweep.

## Open questions

- The ngram leader has no long-range context. Does a non-differentiable long-range channel
  help it at all within 1000 ZO steps? ROSA, counters, and a GRU on top all failed to.
- Every result is at 1000 steps. Does "smaller wins" flip at 10k steps?
- Transformers need their own lr sweep at this config before any conclusion about hard
  attention or step FFNs inside them.

# Overnight run log (autonomous)

Canonical config so far: RMSProp (momentum=none, second_moment=on), lr=2.5e-3,
eps=1.2e-3, bs16_z16, hidden=1024, 1000 steps. Best loss 2.386 (was 2.394 ZO-Adam).

## Results so far
- mom head-to-head: RMSProp 2.386 < Adam 2.399 at lr=2.5e-3. Momentum slightly harmful.
- perturb1: lozo(rank64) 2.402 < gauss 2.438. Polar blocked by bfloat16 QR bug.
- FIXED: polar QR -> cast to float32 (train_zo.py).
- perturb2 running: lozo/polar x rank{16,64,256} x lr{2e-3,2.5e-3,3e-3} (18 runs).
## perturb2 + polar_rank
- lozo: rank256 2.386 best; lozo wants HIGH rank.
- polar: rank16 2.370 (perturb2) but 2.401 (rerun) -> run-to-run noise ~0.03!
- polar_rank: rank8 2.383 best, rank16 2.401, rank32 2.399.
- Conclusion: perturbation distribution effect is small (~0.05) and noisy. gaussian ~2.39-2.44, lozo ~2.40-2.43, polar ~2.37-2.40. Not a big lever at this size.
## fitness + sharedbatch
- Fitness shaping (ES/GRPO) does NOT beat plain MeZO slope: none 2.328 < grpo 2.346 < es 2.377 (shared batch, lr=2.5e-3).
- BIG: shared_batch=True helps MeZO consistently (~0.06): on vs off at z=8/16/32 -> 2.469/2.324/2.216 vs 2.551/2.379/2.300.
- More z_batch helps a lot but confounded with tokens/step (batch fixed 16 -> z=32 = 512 tok/step).
- Best so far: shared+bs16+z32 lr2.5e-3 = 2.216 (vs original 2.394).
## split (compute-normalized, shared on)
- 512 tok/step: bs16_z32 2.228 ~ bs32_z16 2.275 ~ bs8_z64 2.293 < bs64_z8 2.396. Sweet spot batch16-32.
- 1024 tok/step: bs32_z32 2.182 best, bs16_z64 2.208. More tokens/step still helps -> under-trained at 1000 steps.
- Best overall: shared+bs32_z32+lr2.5e-3 = 2.182 (from 2.394 original). ~0.21 total gain.
## steps x lr (HORIZON CONFOUND CONFIRMED)
- Optimal lr DECREASES with training horizon: 1000->2.5e-3, 2000->2e-3, 4000->1.5e-3.
- Loss keeps dropping: 2.16 -> 2.09 -> 2.05. Under-trained at 1000 steps.
- So ALL prior lr conclusions (2.5e-3 optimum, 4e-3 cliff, size scaling) are 1000-step artifacts!
## steps2 (full)
- 6000 steps: lr=1.5e-3 -> 2.039 (best), lr=2e-3 -> 2.084, lr=1e-3 -> 2.474, lr=5e-4 -> 3.017.
- 8000 steps: lr=1.5e-3 -> 2.056, lr=2e-3 -> 2.062 (tied), lr=1e-3 -> 2.443, lr=5e-4 -> 2.998.
- Confirms plateau: lr optimum settles ~1.5-2e-3, does NOT keep decreasing below that.

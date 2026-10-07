# mlsweep feedback from the GradGuess / subspace session (2026-10-04)

One agent session, campaign `GradGuess`, ~46 real runs across 4 mlsweep sweeps
(`tinystories_subspace`, `tinystories_proj`, `tinystories_mix`, plus a first
`tinystories_subspace` that was cancelled and resubmitted). Shared manager, 2 local
GPUs. Goal was to apply "How to guess a gradient" (Singhal et al., 2023) subspace ideas
to the `tinystories_baseline` MeZO/SPSA protocol. The science outcome is in the appendix;
this file is about the tool and the instructions.

## What went well

- **`run --validate`** on every new sweep. Caught duplicate/typo'd combos before
  submission, and it prints the full run list + flag set, which made review easy.
- **`OPTIONS` dimension cross-product.** Once I used it, a 15- or 18-run factorial was a
  ~20-line file. `EXCLUDE(combo)` cleanly removes a control arm that ignores one dim
  (e.g. gaussian ignores `rank`), so `dist={gauss,act,wrow} × rank={…}` gives exactly one
  gaussian run instead of seven. This is the single best time-saver; I did not use it
  until the user pointed it out.
- **`run` without `--stream`** prints the experiment id plus ready-to-paste
  `watch`/`fetch` lines, then returns. No need to block on the submission.
- **`wait --until any-failure --timeout 900`** right after submit is the right "did the
  first wave crash?" primitive. `--until done --timeout 7000` for the rest. Exit codes
  are useful.
- **`logs <run> --experiment <exp>`** gave the full Python traceback for failed runs
  without touching the scratch dirs.
- **`stop --yes` / `retry --failed`** behaved exactly as described.
- **Artifact = working tree.** Uncommitted edits (new perturbation modes, new sweeps)
  shipped to workers with no commit. This is documented in the earlier feedback and it
  held; I never committed just to submit.
- **`MLSWEEP_CAMPAIGN` in the environment** meant most commands needed no `--campaign`.
- **`status`** showing per-GPU util/mem was useful for confirming workers were alive.
- The newer analysis surface is better than the old feedback assumed: `best` prints dims
  inline (`dist=act rank=128`) and supports `--group-by`, `--table`, `--json
  --with-dims`; `metrics` supports `--last`, `--with-dims`, wide CSV. Most of the older
  asks are done.

## What cost time (ordered by impact)

1. **I ran CUDA processes outside mlsweep, on the manager host.** Short `train.py` smoke
   tests and a custom Python diagnostic on GPU 0 collided with the running workers:
   two jobs failed with `CUDA error: CUDA-capable device(s) is/are busy or unavailable`.
   The instructions should state this loudly: *on a host whose GPUs are owned by an
   mlsweep worker, do not start any CUDA process (including 3-step smoke tests and
   diagnostics) — submit a 1-run sweep instead.* It would also help if mlsweep labelled
   that failure mode in `ls`/`wait` (e.g. "device busy — another process holds the GPU")
   instead of leaving it as a generic failed run.
2. **`metrics --tail N` vs `--last`.** I reached for `mlsweep metrics --keys '^loss$'
   --tail 1 --csv` expecting one row per run; `--tail` is "steps per run in the table
   view", so I got the long table. The feature I wanted was `--last --csv` (wide, one row
   per run). The help text is correct but easy to misread. Adding a "did you mean
   `--last`?" hint when `--csv --tail` is used, or mentioning `--last` first in the
   metrics docs, would avoid it.
3. **Hand-enumerating combinations instead of the cross-product.** I wrote out all 15
   `pert` branches in a `flags` dict before being told. The cross-product + `EXCLUDE`
   pattern is buried in `sweep_configuration.md`. Put a worked "control arm excluded from
   a cross product" example at the very top of that doc and in the agent-facing skill.
4. **"Settled cleanly" ≠ "trained to completion."** Runs that trip the `loss_kill`
   guard exit 0 and count as done, and `best` then ranks their min-loss against full
   runs. This session the guard never fired, but for a real sweep it silently mixes
   killed and finished runs. A per-run stop reason in `ls`/`best` (finished /
   kill-guard / crash) would let an agent exclude them. (Flagged before; still true.)
5. **No queue ETA.** `status` gives pending/in-flight counts but no time estimate; an
   18-run sweep took ~40 min on 2 GPUs and I estimated it by hand. Still the top
   convenience ask from the GRT feedback.
6. **Interrupted `wait`.** When the harness aborted a `wait` tool call mid-poll, the
   command printed `Command aborted`. Not mlsweep's fault, but the next agent should
   know `wait` is idempotent and can simply be re-issued; the experiment usually
   continues.
7. **`stop` leaves the experiment paused, and `retry` does not unpause it.** I stopped a
   sweep, then `retry --failed --cancelled` re-queued the jobs — but they sat at
   `jobs_pending=12, jobs_in_flight=0` forever. The missing step was
   `mlsweep unpause <exp>`. `resume` does not help ("Nothing to resume, no failed or
   cancelled jobs"). *Ask: have `retry` auto-unpause, or at minimum print "experiment is
   paused — run `mlsweep unpause <exp>`".*
8. **`retry` silently no-ops once `max_retries` (default 2) is exhausted.** After a few
   transient CUDA device-busy failures, retried jobs hit the cap; `retry --failed` then
   prints only `FAIL  retry <run>` for each, with no reason. The HTTP handler returns
   "max_retries reached" (400), but the CLI drops it. `--dry-run` misleadingly still
   says "would retry". *Ask: surface the reason, and add a `--force` / `--max-retries N`
   escape hatch (the only workaround today is to resubmit a top-up sweep).*

## For the next agent (this codebase)

**Hard rule.** Run experiments only through mlsweep. Do not invoke `train.py` or any
CUDA script directly on the manager host while workers are up; submit a 1-run sweep and
read its log. Do not commit to submit — the artifact is the working tree as-is.

**Workflow**

```sh
mlsweep run sweeps/<file>.py --manager http://localhost:7891 --validate
mlsweep run sweeps/<file>.py --manager http://localhost:7891 --note "<what/why>"   # prints the exp id
mlsweep wait <exp> --until any-failure --timeout 900
mlsweep wait <exp> --until done        --timeout 7000
mlsweep best --experiment <exp> --group-by <dim>          # dims already printed inline
mlsweep metrics --experiment <exp> --keys '^loss$' --last --csv --with-dims
```

Sweep files live in `sweeps/`. The base protocol is whatever `COMMAND` carries; put the
tuned constants in `EXTRA_FLAGS` and only vary genuinely-independent knobs in `OPTIONS`
(cross product + `EXCLUDE`).

**The `tinystories_baseline` protocol** (single reference run, 1000 steps, min train
loss ~2.383, byte GRU `zotitan/tinystories-byte-rnn`):

```
--zo.batch-size 16 --zo.z-batch 16 --zo.eps 0.0009 --zo.base.lr 0.0025
--zo.mom.momentum-method stored_ema --zo.mom.second-moment --training.compile-mode none
```

Background: `findings/FINDINGS.md` (knobs), `findings/OVERNIGHT_REPORT.md`
(lr/horizon/batch findings; best known config is RMSProp, shared_batch, bs32_z32,
lr 1.5e-3, 6000 steps → ~2.04). `shared_batch=True` alone reproducibly helps MeZO by
~0.05–0.08 and showed up again here (gaussian shared-on ≈ 2.371).

## Appendix: the subspace experiment (so it is not repeated)

Added to `PerturbationConfig`: `act`/`wrow` sampling modes, a post-hoc `projection`
knob, and `subspace_mix` (partial restriction). Diagnostic result worth keeping: on a
real TinyStories batch the **true FO gradient's row space is ~98–99% contained in the
top-8 principal directions of the layer's incoming activations**, far better than a
random 8-dim subspace (~8%). So the paper's premise genuinely holds for this GRU.

It still does not help. Hard restricting `z` to that subspace is **monotonically worse**
as rank drops (min loss: r=8 3.08, r=32 2.73, r=128 2.51, r=256 2.52, gaussian 2.40) and
projecting the finished estimate onto the same subspace fails identically (out-of-sample
confirmation that `P_S ĝ = ⟨g, P_S z⟩(P_S z)` is just the restricted-sampling estimator).
Orthogonality is fine (max |cos| ≈ 8e-4, same as gaussian), so the loss is not estimator
redundancy. The likely mechanism: restricting each step's update to the current
activation subspace freezes the walk onto a moving low-rank manifold, so the optimizer can
no longer rotate the subspace itself; isotropic noise is the exploration that changes it.
The paper's gains come from the JVP / per-layer structure (guess a low-dim neuron vector
and form `ΔW = g xᵀ` per layer per example), which a single global SPSA scalar cannot
exploit. The only plausible next attempt would be **per-layer decoupled SPSA** (one
weight block perturbed per scalar, each restricted to its own activation subspace), at
~#blocks× the forward passes — not a tweak.

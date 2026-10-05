# mlsweep feedback from the RestrictedZO campaign (2026-10-01)

One agent session (Claude Code) ran 11 experiments, about 280 runs, in campaign `RestrictedZO`.
The work went: E0 gradient-alignment diagnostics (one run = one estimator × width, with ~100
logged keys), E1 training sweeps, then a head-to-head of the new estimators against the tuned ZO
baseline. It shared one manager with another agent's R-AdaZO campaign (160 jobs). This covers what
worked, what cost time, and what to change, ordered by impact. It builds on `MLSWEEP_ADVICE.md`
(the GRT session). Several of that report's asks now exist (`wait`, `metrics`, campaigns), and they
helped.

## What went well

- **The sweep-file format.** Nested subdims, dict-valued flags and `name: ""` covered every design
  I wanted. Examples: estimator × lr; per-K credit choices (`K36` sweeps 4 credits × 3 lr, `K144`
  sweeps 2 × 2); per-estimator lr pairs built with a dict comprehension. I never had to write a
  launcher script.
- **`--validate`** caught grid mistakes before submission. It is cheap enough to run every time
  (I skipped it once, which caused the duplicate-name failure below).
- **Shipping code from a worktree just worked.** Submitting from `~/git/ZOTitan-rzo` sent that
  tree's code, and the workers picked up the main repo's `.venv` through `remote_dir`. No syncing
  was needed.
- **`wait` with exit codes.** `wait EXP --until any-failure --timeout 900` was a clean "did the
  first wave crash?" check, and `--until done --timeout 6900` fit under the agent's 2-hour
  background limit. This replaced every hand-rolled polling loop.
- **`metrics --csv` / `--json`** gave on-demand access to every logged key without touching the
  DB or the scratch dirs. All analysis went through it.
- **Campaigns + `--note`** kept this project's 11 experiments separate from the other agent's
  campaign, and labeled.
- **`stop --yes`** cleanly killed superseded sweeps when the plan changed.

## What went wrong or cost time

1. **A second `log()` at the same step is silently dropped.** I logged `final_val_loss` /
   `best_val_loss` in a second call with the same step as the last `val_loss`. The DB does
   `INSERT OR IGNORE` on (job, attempt, step), so those keys vanished. `best` then printed
   "0 completed runs" for the sweep's `METRIC`, with no hint why. About 30 minutes went to
   diagnosing it and working around it with `metrics`. Fix: merge same-step dicts. At minimum,
   warn in the logger or worker, and have `best`/`fetch` say "METRIC `final_val_loss` was never
   logged; keys seen: …".
2. **A bad sweep leaves a half-created experiment.** With `"name": None` on a flags-dict dim, every
   run was named `<sweep>_default`. `run` registered the artifact and created the experiment, then
   failed on job insert with a raw `UNIQUE constraint failed: jobs.experiment_id, jobs.run_id`.
   That left a "running" experiment with 0 jobs that I had to `stop`. Fix: check run-name
   uniqueness client-side, the same check `--validate` does, and make experiment creation
   atomic with job insert.
3. **What code a run sees is undocumented.** I didn't know whether the artifact was the git HEAD,
   tracked files, or the working tree, so I committed before every submission "to be safe". The
   user did not want those commits. It turns out `_pack_project` tars the whole directory minus a
   fixed exclude list (and ignores `.gitignore`). State this in the skill/README in one line:
   "the submitted artifact is your working directory as it is now, including uncommitted and
   untracked files; `.git`, `.venv`, `__pycache__`… are excluded". Also print the artifact's file
   count and size at submit time.
4. **There is no "best per group" view.** Every sweep was estimator × lr, and the question is
   always "best lr per estimator, then rank estimators". `best` gives a flat top-N, so I pulled
   CSV and regex-parsed run names back into dims each time (about 8 throwaway Python snippets).
   Ask: `mlsweep best --group-by est`, which prints one row per group (best run, its other dims,
   metric), plus optional `--table est,lr` for a 2-D grid of the metric.
5. **Dims aren't in the metrics output.** Run names encode the dims (`estn1_own_lr3e-4`,
   `K36.cwin8.lr1e-3`), and parsing them back is fragile once names nest. Ask: a `--with-dims`
   flag on `metrics`/`best`/`ls` that adds each dim as a column, from the submitted combo dict
   (`--validate` already prints it).
6. **There is no "last value per run, wide" view.** The common need was one row per run with
   each key's final value. `metrics --tail 1` prints per-run blocks that had to be parsed with
   awk/paste. `--csv` is long-format, which is fine but always needs a pivot. Ask: `metrics --last
   --csv` that prints one row per run and one column per selected key.
7. **`--pivot` handles only one varying index.** The E0 runs logged `s{ckpt}/K{K}/cos/{type}`, a
   2-D sweep inside one run, at different steps. `--pivot` handles one capture group at one
   step, so I used `--json` + Python. Ask: multiple capture groups (rows × columns) and
   "latest value of each key regardless of step". This is still on demand, with no storage.
8. **Queue position and ETA are invisible.** My first submissions sat behind 139 R-AdaZO jobs. I
   estimated the wait myself from done-counts across two `ls` calls (~4.5 h, actually less).
   Long runs (K=144, ~2 h) had no per-run ETA either. Ask: `status`/`ls` show pending jobs ahead
   of mine and a throughput-based ETA. `logger.log(..., total_steps=)` (or a `TOTAL_STEPS`
   convention) would enable per-run progress/ETA; this was already asked for in the GRT report.
9. **Worker disconnects are silent.** The remote SSH worker dropped mid-campaign (`workers=1`
   in `status`, which I only noticed by chance) and came back later. Ask: a `wait`/`watch`
   event and a `status` line ("worker aaron@… disconnected 12 min ago, 2 GPUs"), and have `ls`
   show which worker each running job is on.
10. **A campaign has to be named on every command.** Each Bash call is a fresh shell for an
    agent, so `MLSWEEP_CAMPAIGN` doesn't persist, and every command needed `--campaign
    RestrictedZO`. Forgetting it on `ls <exp>` gives an error, which is correct but noisy. Ask:
    `mlsweep campaign use NAME` writing a per-project default (e.g. `.mlsweep/campaign` in the
    repo root), shown in `status`.
11. **Shared-manager etiquette is undefined.** With another agent's 160 jobs queued, I had to
    decide alone whether `--priority` was acceptable, and chose not to use it across campaigns. A
    per-campaign concurrency share (or showing `--max-concurrent` per campaign in `status`)
    would let several agents share fairly without guessing.

## Smaller things

- `fetch --wait` has no timeout, so an agent with a 2-hour tool limit loses the waiter. `wait
  --timeout` is the right primitive; the skill should say "agents: use `wait --timeout <7000`,
  not `fetch --wait`".
- `best` defaults to `METRIC` from the sweep file. That's good, but when the metric is missing it
  should fall back to listing the available final keys instead of an empty leaderboard.
- `metrics --tail N` prints `(N steps)` inside the per-run header, which breaks naive
  column parsing. It would be one more reason for a strict machine format (point 6).
- Per-host caches (dataset bytes, backprop snapshots for diagnostics) were rebuilt on each
  worker host. This was fine here. A documented `MLSWEEP_CACHE_DIR` shared across runs on a host
  would make the pattern explicit.

## For the next agent (skill text worth adding)

- The artifact is your working tree. Never commit just to submit.
- Log each step's metrics in one `log()` call. A second call with the same step is dropped.
- Run `--validate` on every new sweep file. It catches duplicate run names, which `run` would
  half-create.
- To wait: `wait EXP --until any-failure --timeout 900` right after submit, then
  `wait --until done --timeout 6900` in the background.
- Analysis: `metrics --experiment EXP --keys '^metric$' --tail 1 --csv`. Until `--group-by`
  exists, expect to split run names into dims yourself.

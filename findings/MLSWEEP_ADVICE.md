# Advice from running the GRT sweeps with an agent

This covers the GRT / ZO work of 2026-09-25 to 2026-09-28, run by Claude Code through mlsweep.
It has three parts: how the agent actually used mlsweep, how the collaboration went, and what
to change (in mlsweep, and in how an agent should work). The overnight failures are covered
separately in `MLSWEEP_INCIDENT.md`.

## How the agent used mlsweep

What worked: sweep files with subdims, `--validate`, `--priority`, `pause`/`unpause`,
`cancel`, and `fetch --wait` once it was pointed out.

What didn't:

- The agent never read the agent-facing skill (`mlsweep --help skill`), which exists for
  exactly this. It called `mlsweep logs` without the required `--experiment`, wrote its own
  polling loops until told about `fetch --wait`, and never used `--json`, `--note`,
  `logs --follow`, `best` or `resume`.
- It parsed colored terminal output (`mlsweep ls` through `sed` to strip escape codes) and went
  around mlsweep constantly: reading the live scratch dirs under `/tmp/mlsweep`, querying
  `manager.db` with sqlite, curling `/api/workers`, and writing two analysis scripts
  (`summarize.py`, `trajectories.py`).
- The quantity that mattered was a curve: validation loss as a function of loop count,
  measured at each eval. It changes along two axes (loop count and training step). The dashboard plots a metric against step, trial, or a
  sweep dimension, so the agent logged about 20 `val/nll@rN` keys per run, wrote JSON files
  into the run dir, and did every comparison offline.
- Waiting was fragile. `fetch --wait` outlives the agent's tool timeout, so waits were pushed to
  the background, which left duplicate waiters that fired stale notifications later. The
  failure watcher looked only for `failed`; overnight, runs died without being marked failed
  and the agent didn't notice for 8 hours.
- Control was racy. Cancelling pending runs and then running runs in two calls let jobs start in
  between. `stop --yes` would have done both at once.
- Placement was blind. Long ZO jobs landed on the 12 GB 4070 and took three times as long as on
  the 5090, with no way to steer them.
- Housekeeping: an upload failed at 635 MB with no hint of the cause (a stray checkpoint in the
  repo), about 20 near-duplicate experiment IDs piled up with no record of which replaced
  which, and `fetch` reported "0 completed runs" because diagnostic runs didn't log `loss`.

## How the collaboration went

The user's short corrections were the most valuable input in the whole project. "Learning rate
is the only thing that matters", "z_batch is too low, the curves are flat and spiky", "1 to 8
loops is not enough for FO instability to show up", and "kill the old sweep" each saved hours.
Several came from the user looking at dashboard curves while the agent was reading end-of-run
tables.

The user asked "how long?" twice and "status?" once, which means progress wasn't surfaced
without being asked. And "use less words".

## Advice for an agent running experiments

1. Pilot before scaling. Most of the lost time came from relaunches: cccc to OWT, shared to
   independent batches, 8 loops to 256, sampling depth from 1 to sampling from 4. A 10-minute
   pilot of each arm, with the per-step curves actually looked at, would have caught most of
   them. The agent found a 0.35-nat baseline discrepancy early and kept training on the wrong
   dataset anyway.
2. Test the regime the hypothesis is about first. The shallow-loop phase could never answer
   whether ZO helps at depth; the backprop-versus-depth diagnostic should have run on day one.
3. Keep one live status page with ETAs, so "how long?" is answered before anyone asks.
4. Watch whether runs are advancing. A run that stops making progress is as bad as one that
   fails, and a status check alone won't catch it.
5. Answer direct questions briefly.
6. Read the tool's own agent docs (`mlsweep --help skill`) before the first command.

## How to look at loss curves

Log the right things:

- Log the quantity the question depends on. The ZO training loss averaged over randomly
  sampled loop counts, which mixed several signals and is why it looked spiky. Log per-bucket losses (fixed
  loop counts), and evaluate on fixed windows.
- Log the step-to-step spread (the ZO population's standard deviation, or the batch's), so
  noise shows up as a band rather than as spikes.
- Log compute counters: tokens, FLOPs, wall-clock. ZO against FO only makes sense on a compute
  axis.

Look at them the right way:

- Plot differences against a reference run. The effects here were 0.01 to 0.1 nats on a loss
  near 3. On absolute plots everything looks flat; relative to the base checkpoint the trends
  are obvious.
- Let the x axis be step, tokens, FLOPs or wall-clock, and overlay runs across experiments,
  not just within one.
- Show curve-valued metrics as curves: loss against loop count, one line per eval step, colored
  by step.
- Show smoothed and raw together, so both the trend and the noise are visible.

For the agent itself: render small-multiple PNGs of these views and read them at every
check-in (the agent can view images; it kept reading tables instead), and check each run at
about 5% of its length (is it moving, how noisy is it, how does its slope compare to the
reference) before deciding to continue.

## Changes to mlsweep

Ordered by how much they would have helped here.

1. Agent onboarding. Register the skill with Claude Code so it loads at session start (the
   `slop_guard` script shows the pattern). Fill its gaps: `logs` needs `--experiment`; mention
   `--json`, `--note`, and where live scratch dirs are.
2. An on-demand, read-only view of logged metrics that can turn flat keys into curves,
   for agents that can't use the dashboard. This would have replaced both analysis scripts.
3. (Dropped.) Dashboard x-axis choices and difference-from-a-reference views: the Results tab
   is enough, and differences are easy to compute from `mlsweep metrics --json`.
4. A waiting primitive: `mlsweep wait EXP --until done|any-failure|stalled --timeout N` with
   distinct exit codes, and `watch --events` printing one line per state change.
5. Stall detection: flag runs whose log or metrics haven't moved in N minutes; worker-reported
   liveness; a circuit breaker for GPUs that fail jobs in quick succession (see the incident
   report).
6. Progress and ETA: `logger.log(..., total_steps=)` so `status` and `ls` can show step, rate
   and ETA.
7. Placement hints in the sweep file, such as `GPU_MIN_MEM_GB` or `PREFER = "fastest"`.
8. Experiment lineage: `--group` and `--supersedes`, `ls` hiding superseded or cancelled
   experiments, and comparing runs across experiments in the dashboard.
9. An atomic `cancel --active` covering pending and running runs.
10. A leaderboard metric per sweep (`METRIC = ...`) so `fetch` and `best` rank the right key
    without `OPTIMIZE`.
11. When an upload is too large, list its largest files and point to an ignore file.
12. Plain, uncolored output when stdout isn't a terminal, or `MLSWEEP_JSON=1`.

Item 2 is done, as a read-only view (2026-09-28, uncommitted in `~/git/mlsweep`):
`mlsweep metrics --experiment EXP [runs] --keys REGEX [--pivot] [--step N] [--json | --csv]`
fetches what runs already logged, on demand, and prints it as tables, JSON or CSV. `--pivot` turns flat keys into a
curve using the regex's capture group, so `--keys 'val/nll@r(\d+)' --pivot` prints validation
loss against loop count with one column per run. Nothing is stored or cached, and there are no
new protocol messages or dashboard views.

A first attempt stored curves as a new metric type (a logger API, a database table and a
dashboard tab) and was reverted: curves can be logged several times a second across dozens of
experiments, so they should be derived from existing metrics on demand, and the Results tab
already covers plotting.

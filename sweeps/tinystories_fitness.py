#!/usr/bin/env mlsweep_run
"""Fitness shaping ablation: MeZO slope vs OpenAI-ES (centered_rank) vs GRPO.

A fundamentally different ZO estimator: instead of the +/-eps slope, rank the
z-population's fitness. Requires shared_batch (all z scored on one batch), which
is enabled for every run so the none baseline is apples-to-apples.

3 strategies x 5 lrs = 15 runs. gaussian, RMSProp, bs16_z16, eps=0.0012."""

COMMAND = [
    "python", "train.py",
    "--optimizer", "zo",
    "--objective", "tinystories",
    "--model.model-id", "zotitan/tinystories-byte-rnn",
    "--model.max-seq-len", "256",
    "--training.steps", "1000",
    "--training.no-eval",
    "--training.compile-mode", "none",
    "--zo.mom.momentum-method", "none",
    "--zo.mom.second-moment",
    "--zo.shared-batch",
    "--zo.base.ckpt-every", "1000000",
    "--zo.base.loss-kill-threshold", "10",
    "--zo.base.loss-kill-threshold-patience", "5",
]

GPUS_PER_RUN = 1
EXTRA_FLAGS = ["--zo.batch-size", "16", "--zo.z-batch", "16", "--zo.eps", "0.0012"]

OPTIONS = {
    ".fitness": {
        "flags": {
            "none":   ["--zo.fitness.strategy", "none"],
            "es":     ["--zo.fitness.strategy", "centered_rank"],
            "grpo":   ["--zo.fitness.strategy", "grpo", "--zo.fitness.normalize-std"],
        },
        "name": "fit",
    },
    ".lr": {
        "values": [0.001, 0.0025, 0.005, 0.01, 0.02],
        "flags": "--zo.base.lr",
        "name": "lr",
    },
}

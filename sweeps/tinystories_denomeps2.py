#!/usr/bin/env mlsweep_run
"""denom_eps follow-up: the first sweep showed larger denom_eps shifts the optimal
lr UP and slightly lowers the floor (1e-1 @ lr 4e-3 = 2.439 vs 1e-8 @ 2e-3 = 2.457).
Extend denom_eps higher to find where the trend saturates.

3 denom_eps x 3 lrs = 9 runs. ZO-Adam, bs16_z16, eps 0.0009, 1000 steps.
"""

COMMAND = [
    "python", "train.py",
    "--optimizer", "zo",
    "--objective", "tinystories",
    "--model.model-id", "zotitan/tinystories-byte-rnn",
    "--model.max-seq-len", "256",
    "--training.steps", "1000",
    "--training.no-eval",
    "--training.compile-mode", "none",
    "--zo.mom.momentum-method", "stored_ema",
    "--zo.mom.second-moment",
    "--zo.base.ckpt-every", "1000000",
    "--zo.base.loss-kill-threshold", "10",
    "--zo.base.loss-kill-threshold-patience", "5",
]

GPUS_PER_RUN = 1
EXTRA_FLAGS = ["--zo.batch-size", "16", "--zo.z-batch", "16", "--zo.eps", "0.0009"]

OPTIONS = {
    ".denom_eps": {
        "values": [1e-1, 3e-1, 1e0],
        "flags": "--zo.mom.denom-eps",
        "name": "deps",
    },
    ".lr": {
        "values": [0.004, 0.006, 0.008],
        "flags": "--zo.base.lr",
        "name": "lr",
    },
}

#!/usr/bin/env mlsweep_run
"""Finer lr grid across widths to sharpen the lr-vs-width scaling law. The first
scale sweep used a coarse x2 grid; this refines around the size-specific optima
(2.5e-3 at 10M, 1.25e-3 at 39M).

3 widths x 6 lrs = 18 runs. ZO-Adam, bs16_z16, eps 0.0009, 1000 steps.
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
    ".hidden": {
        "values": [512, 1024, 2048],
        "flags": "--model.hidden",
        "name": "h",
    },
    ".lr": {
        "values": [0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004],
        "flags": "--zo.base.lr",
        "name": "lr",
    },
}

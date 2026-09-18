#!/usr/bin/env mlsweep_run
"""Refined lr x eps grid, focused on the non-failure region.

The previous lr x eps sweep (64 runs) spent most of its budget in the failure
regime: eps >= 0.1 collapses (loss 7 -> 267, acc = random) for every lr, and
lr >= 1.3e-2 barely trains. The useful region was:

    lr  in [1e-3, 6.4e-3],  best ~2.5e-3
    eps in [4e-4, 1e-2],    best ~1.6e-3

This sweep re-grids just that region at higher resolution (still 64 runs),
holding bs16_z16 fixed.
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

# Held fixed: bs16_z16.
EXTRA_FLAGS = ["--zo.batch-size", "16", "--zo.z-batch", "16"]

OPTIONS = {
    ".lr": {
        "values": [
            0.001, 0.0015, 0.002, 0.0025,
            0.003, 0.004, 0.005, 0.006,
        ],
        "flags": "--zo.base.lr",
        "name": "lr",
    },
    ".eps": {
        "values": [
            0.0004, 0.0008, 0.0012, 0.0016,
            0.002, 0.0025, 0.003, 0.004,
        ],
        "flags": "--zo.eps",
        "name": "eps",
    },
}

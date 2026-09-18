#!/usr/bin/env mlsweep_run
"""Settle the canonical optimizer: RMSProp (momentum=none) vs ZO-Adam (stored_ema)
at the known-good lrs, second moment on."""

COMMAND = [
    "python", "train.py",
    "--optimizer", "zo",
    "--objective", "tinystories",
    "--model.model-id", "zotitan/tinystories-byte-rnn",
    "--model.max-seq-len", "256",
    "--training.steps", "1000",
    "--training.no-eval",
    "--training.compile-mode", "none",
    "--zo.mom.second-moment",
    "--zo.base.ckpt-every", "1000000",
    "--zo.base.loss-kill-threshold", "10",
    "--zo.base.loss-kill-threshold-patience", "5",
]

GPUS_PER_RUN = 1
EXTRA_FLAGS = ["--zo.batch-size", "16", "--zo.z-batch", "16", "--zo.eps", "0.0012"]

OPTIONS = {
    ".momentum": {
        "flags": {
            "none": ["--zo.mom.momentum-method", "none"],
            "ema":  ["--zo.mom.momentum-method", "stored_ema"],
        },
        "name": "mom",
    },
    ".lr": {
        "values": [0.002, 0.0025, 0.003],
        "flags": "--zo.base.lr",
        "name": "lr",
    },
}

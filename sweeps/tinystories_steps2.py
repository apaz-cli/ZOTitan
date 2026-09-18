#!/usr/bin/env mlsweep_run
"""Longer horizon, lower lr: the optimal lr keeps decreasing with steps
(2.5e-3@1k -> 2e-3@2k -> 1.5e-3@4k). Does it plateau near 1e-3 or keep going?

2 horizons x 4 lrs = 8 runs. shared on, bs32_z32, RMSProp, gaussian, eps=0.0012."""

COMMAND = [
    "python", "train.py",
    "--optimizer", "zo",
    "--objective", "tinystories",
    "--model.model-id", "zotitan/tinystories-byte-rnn",
    "--model.max-seq-len", "256",
    "--training.no-eval",
    "--training.compile-mode", "none",
    "--zo.mom.momentum-method", "none",
    "--zo.mom.second-moment",
    "--zo.shared-batch",
    "--zo.batch-size", "32",
    "--zo.z-batch", "32",
    "--zo.eps", "0.0012",
    "--zo.base.ckpt-every", "1000000",
    "--zo.base.loss-kill-threshold", "10",
    "--zo.base.loss-kill-threshold-patience", "5",
]

GPUS_PER_RUN = 1

OPTIONS = {
    ".steps": {
        "values": [6000, 8000],
        "flags": "--training.steps",
        "name": "steps",
    },
    ".lr": {
        "values": [0.0005, 0.001, 0.0015, 0.002],
        "flags": "--zo.base.lr",
        "name": "lr",
    },
}

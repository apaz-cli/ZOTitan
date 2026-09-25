#!/usr/bin/env mlsweep_run
"""Held-out test of the lr transfer rule: predict lr_opt for hidden=1536 (an unseen
size) from the fitted law (lr ~ 1/width, ~N^-0.4) and verify by sweeping lr around
the prediction (~1.7e-3).

5 lrs = 5 runs. ZO-Adam, bs16_z16, eps 0.001, 1000 steps.
"""

COMMAND = [
    "python", "train.py",
    "--optimizer", "zo",
    "--objective", "tinystories",
    "--model.model-id", "zotitan/tinystories-byte-rnn",
    "--model.max-seq-len", "256",
    "--model.hidden", "1536",
    "--training.steps", "1000",
    "--training.no-eval",
    "--training.compile-mode", "none",
    "--zo.mom.momentum-method", "stored_ema",
    "--zo.mom.second-moment",
    "--zo.eps", "0.001",
    "--zo.base.ckpt-every", "1000000",
    "--zo.base.loss-kill-threshold", "10",
    "--zo.base.loss-kill-threshold-patience", "5",
]

GPUS_PER_RUN = 1
EXTRA_FLAGS = ["--zo.batch-size", "16", "--zo.z-batch", "16"]

OPTIONS = {
    ".lr": {
        "values": [0.00125, 0.0016, 0.002, 0.0025, 0.003],
        "flags": "--zo.base.lr",
        "name": "lr",
    },
}

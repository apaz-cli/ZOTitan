#!/usr/bin/env mlsweep_run
"""eps transfer across size: is the optimal eps size-independent (scale-free) as
the SPSA cancellation predicts? Sweep eps at two widths, at each width's optimal lr.

2 widths x 3 eps = 6 runs. ZO-Adam, bs16_z16, 1000 steps.
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
EXTRA_FLAGS = ["--zo.batch-size", "16", "--zo.z-batch", "16"]

OPTIONS = {
    ".hidden": {
        "flags": {
            "10M":  ["--model.hidden", "1024", "--zo.base.lr", "0.0025"],
            "154M": ["--model.hidden", "4096", "--zo.base.lr", "0.00125"],
        },
        "name": "h",
    },
    ".eps": {
        "values": [0.0003, 0.001, 0.003],
        "flags": "--zo.eps",
        "name": "eps",
    },
}

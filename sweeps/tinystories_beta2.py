#!/usr/bin/env mlsweep_run
"""beta2 sweep: second-moment EMA decay. beta2 controls how fast the denominator
tracks the gradient scale; a faster (lower) beta2 may adapt better and lower the
floor. Sweep beta2 against lr.

4 beta2 x 3 lrs = 12 runs. ZO-Adam, bs16_z16, eps 0.0009, 1000 steps.
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
    ".beta2": {
        "values": [0.9, 0.99, 0.999, 0.9999],
        "flags": "--zo.mom.beta2",
        "name": "b2",
    },
    ".lr": {
        "values": [0.002, 0.003, 0.004],
        "flags": "--zo.base.lr",
        "name": "lr",
    },
}

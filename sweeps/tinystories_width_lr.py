#!/usr/bin/env mlsweep_run
"""Width x lr scaling sweep: does the optimal lr shift with network width?

Scales the GRU's hidden size (width) only, counting non-embedding/head params:

    hidden 1632 ->  25M body     hidden 2848 ->  75M body
    hidden 2304 ->  50M body     hidden 3296 -> 100M body

lr stays in the range known to work at the base model (hidden 1024, ~10M body):
[1e-3, 2.5e-3]. eps is held at 0.0012 and bs16_z16 fixed.

4 widths x 4 lrs = 16 runs.
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

# Held fixed: bs16_z16 and eps = 0.0012 (best from the lr x eps sweep).
EXTRA_FLAGS = ["--zo.batch-size", "16", "--zo.z-batch", "16", "--zo.eps", "0.0012"]

OPTIONS = {
    ".width": {
        "flags": {
            "25m":  ["--model.hidden", "1632"],
            "50m":  ["--model.hidden", "2304"],
            "75m":  ["--model.hidden", "2848"],
            "100m": ["--model.hidden", "3296"],
        },
        "name": "w",
    },
    ".lr": {
        "values": [0.001, 0.0015, 0.002, 0.0025],
        "flags": "--zo.base.lr",
        "name": "lr",
    },
}

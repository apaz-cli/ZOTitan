#!/usr/bin/env mlsweep_run
"""Depth x lr ablation: does the optimal lr shift with network depth?

Scales the GRU layer count only (hidden width fixed at 1024). The first layer is
smaller than the rest (input is d_model=256 rather than hidden=1024), so the
actual non-embedding/head body sizes are:

    num_layers 1 ->  3.9M   (labeled 5m)
    num_layers 2 -> 10.2M   (labeled 10m)
    num_layers 3 -> 16.5M   (labeled 15m)
    num_layers 4 -> 22.8M   (labeled 20m)
    num_layers 5 -> 29.1M   (labeled 25m)

lr stays in the known-good range [1e-3, 2.5e-3]. eps = 0.0012, bs16_z16 fixed.

5 depths x 4 lrs = 20 runs.
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

# Held fixed: bs16_z16 and eps = 0.0012.
EXTRA_FLAGS = ["--zo.batch-size", "16", "--zo.z-batch", "16", "--zo.eps", "0.0012"]

OPTIONS = {
    ".depth": {
        "flags": {
            "5m":  ["--model.num-layers", "1"],
            "10m": ["--model.num-layers", "2"],
            "15m": ["--model.num-layers", "3"],
            "20m": ["--model.num-layers", "4"],
            "25m": ["--model.num-layers", "5"],
        },
        "name": "d",
    },
    ".lr": {
        "values": [0.001, 0.0015, 0.002, 0.0025],
        "flags": "--zo.base.lr",
        "name": "lr",
    },
}

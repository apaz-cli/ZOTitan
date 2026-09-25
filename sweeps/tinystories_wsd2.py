#!/usr/bin/env mlsweep_run
"""Clean WSD sweep at the DEFAULT optimizer (denom_eps=1e-8, beta2=0.999), lr around
the 2.5e-3 optimum. Tests whether cosine decay (and a bit of warmup) beats constant
lr at 1000 steps.

4 decay_frac x 3 lrs = 12 runs. ZO-Adam, bs16_z16, eps 0.0009, 1000 steps.
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
    ".decay_frac": {
        "values": [0.0, 0.3, 0.5, 0.8],
        "flags": "--zo.base.lr-wsd.decay-frac",
        "name": "df",
    },
    ".lr": {
        "values": [0.002, 0.0025, 0.003],
        "flags": "--zo.base.lr",
        "name": "lr",
    },
}

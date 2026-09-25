#!/usr/bin/env mlsweep_run
"""MoE: how does a top-k mixture-of-experts FFN block shift the optimal lr and ZO
eps vs the dense baseline (num_experts=1)?

3 num_experts x 3 lrs x 2 eps = 18 runs. ZO-Adam, bs16_z16, top_k=2, 1000 steps.
"""

COMMAND = [
    "python", "train.py",
    "--optimizer", "zo",
    "--objective", "tinystories",
    "--model.model-id", "zotitan/tinystories-byte-rnn",
    "--model.max-seq-len", "256",
    "--model.top-k", "2",
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
    ".num_experts": {
        "values": [1, 4, 8],
        "flags": "--model.num-experts",
        "name": "e",
    },
    ".lr": {
        "values": [0.0015, 0.0025, 0.004],
        "flags": "--zo.base.lr",
        "name": "lr",
    },
    ".eps": {
        "values": [0.001, 0.003],
        "flags": "--zo.eps",
        "name": "eps",
    },
}

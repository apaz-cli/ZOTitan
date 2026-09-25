#!/usr/bin/env mlsweep_run
"""Bigger MoE experiment: MoE's benefit is at scale (more capacity per active FLOP),
so test at larger widths whether adding experts now helps the loss. Residual MoE FFN
(fixed). lr uses the transferred ~1/width values, eps=1e-3.

2 widths x 4 num_experts x 3 lrs = 24 runs. ZO-Adam, bs16_z16, top_k=2, 1000 steps.
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
    "--zo.eps", "0.001",
    "--zo.base.ckpt-every", "1000000",
    "--zo.base.loss-kill-threshold", "10",
    "--zo.base.loss-kill-threshold-patience", "5",
]

GPUS_PER_RUN = 1
EXTRA_FLAGS = ["--zo.batch-size", "16", "--zo.z-batch", "16"]

OPTIONS = {
    ".hidden": {
        "values": [2048, 4096],
        "flags": "--model.hidden",
        "name": "h",
    },
    ".num_experts": {
        "values": [1, 4, 8, 16],
        "flags": "--model.num-experts",
        "name": "e",
    },
    ".lr": {
        "values": [0.001, 0.0015, 0.002],
        "flags": "--zo.base.lr",
        "name": "lr",
    },
}

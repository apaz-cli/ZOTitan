#!/usr/bin/env mlsweep_run
"""fp8 weight quantization: how does storing weights on the torch float8 grid shift
the optimal lr and ZO eps?

Key prior: with fp8 weights, eps must exceed the quantization step (~0.1 relative
for e4m3) or the perturbation ±εz is rounded away and proj_grad = 0 (verified: eps
0.0009 gives proj_grad 0 under fp8). So eps is swept much higher than bf16.

2 quant formats x 3 eps x 3 lrs = 18 runs. ZO-Adam, bs16_z16, 1000 steps.
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
    ".quant": {
        "flags": {
            "fp8e4m3": ["--zo.weight-quant", "fp8_e4m3"],
            "fp8e5m2": ["--zo.weight-quant", "fp8_e5m2"],
        },
        "name": "q",
    },
    ".eps": {
        "values": [0.03, 0.1, 0.3],
        "flags": "--zo.eps",
        "name": "eps",
    },
    ".lr": {
        "values": [0.001, 0.003, 0.01],
        "flags": "--zo.base.lr",
        "name": "lr",
    },
}

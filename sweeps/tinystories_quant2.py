#!/usr/bin/env mlsweep_run
"""fp8 QAT-style sweep: quantize weights ONCE, perturb the bf16 master on top (the
mode that works). Measures the REAL lr/eps shift under fp8 weight storage — expected
≈ bf16, since the perturbation lives in bf16.

2 quant formats x 3 lrs x 2 eps = 12 runs. ZO-Adam, bs16_z16, 1000 steps.
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
    "--zo.weight-quant-mode", "once",
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
    ".lr": {
        "values": [0.002, 0.0025, 0.003],
        "flags": "--zo.base.lr",
        "name": "lr",
    },
    ".eps": {
        "values": [0.001, 0.003],
        "flags": "--zo.eps",
        "name": "eps",
    },
}

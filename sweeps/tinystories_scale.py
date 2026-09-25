#!/usr/bin/env mlsweep_run
"""Width scaling for the lr transfer law: sweep width x lr, fit lr_opt(N) ~ N^-alpha.

Widths (hidden): 512/1024/2048/4096 -> body ~2.6M/10.2M/41M/164M. lr grid spans the
expected ~N^-0.5 optima (2.5e-3 at 10M). eps fixed (scale-free). Default ZO-Adam.

4 widths x 4 lrs = 16 runs. bs16_z16, eps 0.0009, 1000 steps.
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
    ".hidden": {
        "values": [512, 1024, 2048, 4096],
        "flags": "--model.hidden",
        "name": "h",
    },
    ".lr": {
        "values": [0.0006, 0.00125, 0.0025, 0.005],
        "flags": "--zo.base.lr",
        "name": "lr",
    },
}

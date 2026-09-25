#!/usr/bin/env mlsweep_run
"""denom_eps sweep: soft sign-SGD. The second-moment denominator (default 1e-8)
makes MeZO a fixed-step sign-SGD; a larger denom_eps lets the step shrink with the
gradient near the minimum, which should lower the loss floor. Sweep denom_eps
against lr (higher denom_eps may tolerate/need higher lr).

5 denom_eps x 3 lrs = 15 runs. ZO-Adam, bs16_z16, eps 0.0009, 1000 steps.
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
    ".denom_eps": {
        "values": [1e-8, 1e-4, 1e-3, 1e-2, 1e-1],
        "flags": "--zo.mom.denom-eps",
        "name": "deps",
    },
    ".lr": {
        "values": [0.002, 0.003, 0.004],
        "flags": "--zo.base.lr",
        "name": "lr",
    },
}

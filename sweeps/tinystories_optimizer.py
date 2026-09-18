#!/usr/bin/env mlsweep_run
"""Optimizer-structure ablation: is ZO-Adam actually the right optimizer?

We've used stored_ema momentum + second moment (ZO-Adam) throughout, and the
clipping result showed the second moment divides the gradient magnitude out of
the update — i.e. ZO-Adam here behaves like sign-MeZO with a fixed step lr.
This sweeps the optimizer structure itself against lr:

    momentum    none        (plain MeZO)
                stored_ema  (ZO-momentum)
    second      off / on    (on = Adam/RMSProp denominator)

Four configs, each over a wide log lr sweep (plain MeZO wants ~1e-4, ZO-Adam
~2.5e-3, so the scales differ by ~100x).

2 momentum x 2 second x 7 lr = 28 runs.
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
    "--zo.base.ckpt-every", "1000000",
    "--zo.base.loss-kill-threshold", "10",
    "--zo.base.loss-kill-threshold-patience", "5",
]

GPUS_PER_RUN = 1

# Held fixed: bs16_z16 and eps = 0.0012.
EXTRA_FLAGS = ["--zo.batch-size", "16", "--zo.z-batch", "16", "--zo.eps", "0.0012"]

OPTIONS = {
    ".momentum": {
        "flags": {
            "none": ["--zo.mom.momentum-method", "none"],
            "ema":  ["--zo.mom.momentum-method", "stored_ema"],
        },
        "name": "mom",
    },
    ".second": {
        "flags": {
            "off": [],
            "on":  ["--zo.mom.second-moment"],
        },
        "name": "sec",
    },
    ".lr": {
        "values": ["1e-5", "3e-5", "1e-4", "3e-4", "1e-3", "3e-3", "1e-2"],
        "flags": "--zo.base.lr",
        "name": "lr",
    },
}

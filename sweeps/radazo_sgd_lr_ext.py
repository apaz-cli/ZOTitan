#!/usr/bin/env mlsweep_run
"""Extends radazo_optimizers.py upward for the non-adaptive rules.

MeZO and MeZO-momentum were still improving at that sweep's top lr (5.6e-3), so their
optimum wasn't bracketed. Same settings, higher lrs.

2 rules × 3 lr = 6 runs.
"""

COMMAND = [
    "python", "train.py",
    "--optimizer", "zo",
    "--objective", "tinystories",
    "--model.model-id", "zotitan/tinystories-byte-rnn",
    "--model.max-seq-len", "256",
    "--training.steps", "1000",
    "--training.compile-mode", "none",
    "--zo.base.ckpt-every", "1000000",
    "--zo.base.loss-kill-threshold", "10",
    "--zo.base.loss-kill-threshold-patience", "5",
]

GPUS_PER_RUN = 1
METRIC = "ppl"
GOAL = "minimize"

# Known-good estimator settings (findings/FINDINGS.md); eps is a flat plateau.
EXTRA_FLAGS = ["--zo.batch-size", "16", "--zo.z-batch", "16", "--zo.eps", "0.0012",
               "--zo.mom.beta1", "0.9"]

EMA      = ["--zo.mom.momentum-method", "stored_ema"]
SECOND   = ["--zo.mom.second-moment"]
FROM_M   = ["--zo.mom.second-moment-source", "momentum"]
NO_BC    = ["--zo.mom.no-bias-correction"]

BETA2 = {"values": ["0.99", "0.999"], "flags": "--zo.mom.beta2", "name": "b2"}

OPTIONS = {
    ".opt": {
        ".mezo":    {"flags": []},
        ".mezomom": {"flags": EMA},
    },
    ".lr": {
        "values": ["1e-2", "1.8e-2", "3.2e-2"],
        "flags": "--zo.base.lr",
        "name": "lr",
    },
}

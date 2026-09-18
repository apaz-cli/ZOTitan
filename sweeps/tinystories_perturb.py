#!/usr/bin/env mlsweep_run
"""Perturbation distribution ablation: gaussian vs LOZO vs ZO-Muon (polar).

The perturbation is the one knob in ZO we've never touched. gaussian is standard
MeZO; low_rank is LOZO; polar is ZO-Muon (polar-orthogonal factors, known to be
a big ZO win). rank=64 fixed for the factorized ones.

3 distributions x 4 lrs = 12 runs. ZO-Adam, eps=0.0012, bs16_z16.
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

EXTRA_FLAGS = ["--zo.batch-size", "16", "--zo.z-batch", "16", "--zo.eps", "0.0012"]

OPTIONS = {
    ".dist": {
        "flags": {
            "gauss": ["--zo.perturbation.distribution", "gaussian"],
            "lozo":  ["--zo.perturbation.distribution", "low_rank", "--zo.perturbation.rank", "64"],
            "polar": ["--zo.perturbation.distribution", "polar", "--zo.perturbation.rank", "64"],
        },
        "name": "dist",
    },
    ".lr": {
        "values": [0.0015, 0.002, 0.0025, 0.003],
        "flags": "--zo.base.lr",
        "name": "lr",
    },
}

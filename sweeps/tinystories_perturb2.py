#!/usr/bin/env mlsweep_run
"""Perturbation refinement: LOZO vs ZO-Muon (polar) over rank and lr.

First pass showed LOZO(rank64) ~ 2.40 vs gaussian ~2.44. Polar (ZO-Muon) was
blocked by a bfloat16 QR bug (now fixed). Here we sweep rank for both factorized
distributions, with the canonical optimizer (RMSProp: momentum=none + second
moment), which just won the head-to-head at 2.386.

2 dists x 3 ranks x 3 lrs = 18 runs.
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
    "--zo.mom.momentum-method", "none",
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
            "lozo":  ["--zo.perturbation.distribution", "low_rank"],
            "polar": ["--zo.perturbation.distribution", "polar"],
        },
        "name": "dist",
    },
    ".rank": {
        "values": [16, 64, 256],
        "flags": "--zo.perturbation.rank",
        "name": "r",
    },
    ".lr": {
        "values": [0.002, 0.0025, 0.003],
        "flags": "--zo.base.lr",
        "name": "lr",
    },
}

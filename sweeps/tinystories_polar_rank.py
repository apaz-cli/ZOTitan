#!/usr/bin/env mlsweep_run
"""ZO-Muon (polar) rank refinement. polar rank=16 hit 2.370, beating gaussian and
lozo; lower rank looks better (16 > 64 > 256). Sweep rank down to 2.

5 ranks x 3 lrs = 15 runs. RMSProp, eps=0.0012, bs16_z16."""

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
    "--zo.perturbation.distribution", "polar",
    "--zo.base.ckpt-every", "1000000",
    "--zo.base.loss-kill-threshold", "10",
    "--zo.base.loss-kill-threshold-patience", "5",
]

GPUS_PER_RUN = 1
EXTRA_FLAGS = ["--zo.batch-size", "16", "--zo.z-batch", "16", "--zo.eps", "0.0012"]

OPTIONS = {
    ".rank": {
        "values": [2, 4, 8, 16, 32],
        "flags": "--zo.perturbation.rank",
        "name": "r",
    },
    ".lr": {
        "values": [0.002, 0.0025, 0.003],
        "flags": "--zo.base.lr",
        "name": "lr",
    },
}

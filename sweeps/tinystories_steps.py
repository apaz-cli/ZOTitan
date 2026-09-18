#!/usr/bin/env mlsweep_run
"""Training-horizon x lr: the big confound. All results so far are at 1000 steps
where loss is still dropping (under-trained). Does the optimal lr shift at longer
horizon?

4 lrs x 3 horizons = 12 runs. shared on, bs32_z32, RMSProp, gaussian, eps=0.0012."""

COMMAND = [
    "python", "train.py",
    "--optimizer", "zo",
    "--objective", "tinystories",
    "--model.model-id", "zotitan/tinystories-byte-rnn",
    "--model.max-seq-len", "256",
    "--training.no-eval",
    "--training.compile-mode", "none",
    "--zo.mom.momentum-method", "none",
    "--zo.mom.second-moment",
    "--zo.shared-batch",
    "--zo.batch-size", "32",
    "--zo.z-batch", "32",
    "--zo.eps", "0.0012",
    "--zo.base.ckpt-every", "1000000",
    "--zo.base.loss-kill-threshold", "10",
    "--zo.base.loss-kill-threshold-patience", "5",
]

GPUS_PER_RUN = 1

OPTIONS = {
    ".steps": {
        "values": [1000, 2000, 4000],
        "flags": "--training.steps",
        "name": "steps",
    },
    ".lr": {
        "values": [0.0015, 0.002, 0.0025, 0.003],
        "flags": "--zo.base.lr",
        "name": "lr",
    },
}

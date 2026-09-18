#!/usr/bin/env mlsweep_run
"""shared_batch confirmation: the fitness sweep's 'none' baseline used
shared_batch=True and got 2.328 vs 2.386 non-shared — but the code docs claim
shared adds a noise floor for MeZO. Test shared on/off against z_batch and lr.

2 shared x 3 z_batch x 3 lr = 18 runs. gaussian, RMSProp, bs=16, eps=0.0012."""

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
EXTRA_FLAGS = ["--zo.batch-size", "16", "--zo.eps", "0.0012"]

OPTIONS = {
    ".shared": {
        "flags": {
            "off": [],
            "on":  ["--zo.shared-batch"],
        },
        "name": "sh",
    },
    ".z_batch": {
        "values": [8, 16, 32],
        "flags": "--zo.z-batch",
        "name": "z",
    },
    ".lr": {
        "values": [0.002, 0.0025, 0.003],
        "flags": "--zo.base.lr",
        "name": "lr",
    },
}

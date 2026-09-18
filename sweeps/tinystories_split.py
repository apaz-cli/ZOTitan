#!/usr/bin/env mlsweep_run
"""Compute-normalized split: at fixed 512 tokens/step, does more z (smaller batch)
win once shared_batch is on? Isolates z-averaging from tokens/step.

6 splits x 3 lrs = 18 runs. shared_batch on, gaussian, RMSProp, eps=0.0012."""

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
    "--zo.shared-batch",
    "--zo.base.ckpt-every", "1000000",
    "--zo.base.loss-kill-threshold", "10",
    "--zo.base.loss-kill-threshold-patience", "5",
]

GPUS_PER_RUN = 1
EXTRA_FLAGS = ["--zo.eps", "0.0012"]

OPTIONS = {
    ".split": {
        "flags": {
            "bs8_z64":   ["--zo.batch-size", "8",  "--zo.z-batch", "64"],
            "bs16_z32":  ["--zo.batch-size", "16", "--zo.z-batch", "32"],
            "bs32_z16":  ["--zo.batch-size", "32", "--zo.z-batch", "16"],
            "bs64_z8":   ["--zo.batch-size", "64", "--zo.z-batch", "8"],
            "bs16_z64":  ["--zo.batch-size", "16", "--zo.z-batch", "64"],
            "bs32_z32":  ["--zo.batch-size", "32", "--zo.z-batch", "32"],
        },
        "name": "split",
    },
    ".lr": {
        "values": [0.002, 0.0025, 0.003],
        "flags": "--zo.base.lr",
        "name": "lr",
    },
}

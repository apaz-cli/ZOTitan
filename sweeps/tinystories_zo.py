#!/usr/bin/env mlsweep_run
"""ZO training of the tiny byte-level GRU LM on TinyStories.

Uses ZO-Adam (stored-EMA momentum + second moment). The two knobs that matter
most for ZO variance are swept here: the number of perturbations per step
(z_batch) and the learning rate. batch_size is left moderate — more
perturbations beat a bigger batch for the same forward-pass budget.
"""

COMMAND = [
    "python", "train.py",
    "--optimizer", "zo",
    "--objective", "tinystories",
    "--model.model-id", "zotitan/tinystories-byte-rnn",
    "--model.max-seq-len", "256",
    "--training.no-eval",
    "--training.compile-mode", "none",
    "--zo.eps", "1e-4",
    "--zo.batch-size", "32",
    "--zo.mom.momentum-method", "stored_ema",
    "--zo.mom.second-moment",
    "--zo.base.ckpt-every", "1000000",
]

GPUS_PER_RUN = 1

OPTIONS = {
    ".steps": {
        "values": [2000],
        "flags": "--training.steps",
        "name": "steps",
    },
    ".z": {
        "values": [4, 8],
        "flags": "--zo.z-batch",
        "name": "z",
    },
    ".lr": {
        "values": [1e-3, 3e-3],
        "flags": "--zo.base.lr",
        "name": "lr",
    },
}

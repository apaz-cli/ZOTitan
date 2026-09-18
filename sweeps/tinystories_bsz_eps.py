#!/usr/bin/env mlsweep_run
"""Grid search over batch_split x eps to map their interaction.

lr is fixed at 0.0025 (the best value from tinystories_megasweep). The eps
range extends past the old 1e-3 ceiling, which the best megasweep run hit.

    4 batch splits  x  16 eps values  =  64 runs.
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

# lr retained from the best megasweep run (rounded).
EXTRA_FLAGS = ["--zo.base.lr", "0.0025"]

OPTIONS = {
    ".batch_split": {
        # batch_size x z_batch = 256 for every branch -> equal data per step.
        "flags": {
            "bs8_z32":  ["--zo.batch-size", "8",  "--zo.z-batch", "32"],
            "bs16_z16": ["--zo.batch-size", "16", "--zo.z-batch", "16"],
            "bs32_z8":  ["--zo.batch-size", "32", "--zo.z-batch", "8"],
            "bs64_z4":  ["--zo.batch-size", "64", "--zo.z-batch", "4"],
        },
        "name": "bsz",
    },
    ".eps": {
        # log-spaced, extends past the previous 1e-3 upper bound.
        "values": [
            0.0001, 0.000125, 0.00016, 0.0002,
            0.00025, 0.00032, 0.0004, 0.0005,
            0.00063, 0.0008, 0.001, 0.00125,
            0.0016, 0.002, 0.0025, 0.003,
        ],
        "flags": "--zo.eps",
        "name": "eps",
    },
}

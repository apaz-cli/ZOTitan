#!/usr/bin/env mlsweep_run
"""Grid search over lr x eps to check whether they interact / scale together.

batch_size and z_batch are held at 16 (bs16_z16, the best split). Both lr and
eps are log-spaced powers of two over a wide range (128x each):

    lr  : 2e-4  .. 2.56e-2   (8 values)
    eps : 1e-3  .. 3.0       (8 values, big — eps is scale-free to first
                               order in the SPSA estimate, so we probe large
                               values to find where the curvature bias bites)

8 x 8 = 64 runs. The question: does the optimal eps shift with lr (and vice
versa), or are they independent?
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

# Held fixed: bs16_z16.
EXTRA_FLAGS = ["--zo.batch-size", "16", "--zo.z-batch", "16"]

OPTIONS = {
    ".lr": {
        "values": [
            0.0002, 0.0004, 0.0008, 0.0016,
            0.0032, 0.0064, 0.0128, 0.0256,
        ],
        "flags": "--zo.base.lr",
        "name": "lr",
    },
    ".eps": {
        "values": [
            0.001, 0.003, 0.01, 0.03,
            0.1, 0.3, 1.0, 3.0,
        ],
        "flags": "--zo.eps",
        "name": "eps",
    },
}

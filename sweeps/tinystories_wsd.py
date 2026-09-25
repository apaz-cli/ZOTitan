#!/usr/bin/env mlsweep_run
"""WSD lr schedule: cosine decay of lr to 0 over decay_frac of training, with the
best optimizer shape so far (denom_eps=1e-1, beta2=0.9). decay_frac=0 = constant
control. The horizon finding (optimal lr decreases with steps) predicts decay helps.

4 decay_frac x 3 lrs = 12 runs. ZO-Adam, bs16_z16, eps 0.0009, 1000 steps.
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
    "--zo.mom.denom-eps", "0.1",
    "--zo.mom.beta2", "0.9",
    "--zo.base.ckpt-every", "1000000",
    "--zo.base.loss-kill-threshold", "10",
    "--zo.base.loss-kill-threshold-patience", "5",
]

GPUS_PER_RUN = 1
EXTRA_FLAGS = ["--zo.batch-size", "16", "--zo.z-batch", "16", "--zo.eps", "0.0009"]

OPTIONS = {
    ".decay_frac": {
        "values": [0.0, 0.3, 0.5, 0.8],
        "flags": "--zo.base.lr-wsd.decay-frac",
        "name": "df",
    },
    ".lr": {
        "values": [0.004, 0.006, 0.008],
        "flags": "--zo.base.lr",
        "name": "lr",
    },
}

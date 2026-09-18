#!/usr/bin/env mlsweep_run
"""Bayesian hyperparameter sweep for MeZO-Adam on the tiny byte-level GRU LM.

Optimizes three things with TPE (optuna, via mlsweep):

  - lr            (log-uniform, continuous)
  - batch/z split (discrete; batch_size × z_batch = 256, so every run sees the
                   same number of tokens per step and they're comparable)
  - eps           (log-uniform, continuous — the MeZO perturbation scale)
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

OPTIMIZE = {
    "method": "bayes",
    "metric": "loss",      # best (minimum) training loss logged via MLSweepLogger
    "goal": "minimize",
    "budget": 200,         # total successful runs
    "n_initial": 40,       # large warmup before TPE takes over
}

OPTIONS = {
    ".lr": {
        "distribution": "log_uniform",
        "min": 1e-4,
        "max": 1e-2,
        "flags": "--zo.base.lr",
        "name": "lr",
    },
    ".batch_split": {
        # batch_size × z_batch = 256 for every branch → equal data per step.
        "flags": {
            "bs8_z32":  ["--zo.batch-size", "8",  "--zo.z-batch", "32"],
            "bs16_z16": ["--zo.batch-size", "16", "--zo.z-batch", "16"],
            "bs32_z8":  ["--zo.batch-size", "32", "--zo.z-batch", "8"],
            "bs64_z4":  ["--zo.batch-size", "64", "--zo.z-batch", "4"],
        },
        "name": "bsz",
    },
    ".eps": {
        "distribution": "log_uniform",
        "min": 1e-5,
        "max": 1e-3,
        "flags": "--zo.eps",
        "name": "eps",
    },
}

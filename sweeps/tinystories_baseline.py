#!/usr/bin/env mlsweep_run
"""Single-run baseline: the rounded best config from tinystories_megasweep.

Best run: tinystories_megasweep_bayes_0105 (min train loss 2.383, acc 0.375).

Rounded hyperparameters:
    lr  = 0.0025496020987436935  ->  0.0025
    bsz = bs16_z16                ->  batch-size 16, z-batch 16
    eps = 0.0009076076759033956   ->  0.0009
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

# Rounded baseline values (appended to every run).
EXTRA_FLAGS = [
    "--zo.base.lr", "0.0025",
    "--zo.batch-size", "16",
    "--zo.z-batch", "16",
    "--zo.eps", "0.0009",
]

OPTIONS = {}  # single run -> name will be "tinystories_baseline_default"

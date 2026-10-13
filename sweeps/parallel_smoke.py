#!/usr/bin/env mlsweep_run
"""Smoke test of each data-parallel layout on 2 GPUs.

SET_DIST_ENV makes mlsweep start one process per GPU with RANK / WORLD_SIZE /
LOCAL_RANK / MASTER_* set. --parallel.dp must equal GPUS_PER_RUN.
"""

COMMAND = [
    "python", "train.py",
    "--objective", "scijudge",
    "--training.steps", "5",
    "--training.no-eval",
    "--training.compile-mode", "none",
    "--zo.base.ckpt-every", "1000000",
    "--fo.base.ckpt-every", "1000000",
]

GPUS_PER_RUN = 2
SET_DIST_ENV = True

OPTIONS = {
    ".layout": {
        "name": "layout",
        "flags": {
            "zo_dp2_pairs": ["--parallel.dp", "2", "--zo.z-batch", "4", "--zo.batch-size", "2"],
            "zo_dp2_batch": ["--parallel.dp", "2", "--zo.z-batch", "1", "--zo.batch-size", "4"],
            "fo_dp2":       ["--parallel.dp", "2", "--optimizer", "fo", "--fo.batch-size", "4"],
        },
    },
}

#!/usr/bin/env mlsweep_run
# Evaluate checkpoints on SciJudgeBench splits.
# Add checkpoint paths to .checkpoint as training runs complete.

COMMAND      = ["bash", "run.sh", "eval.py"]
GPUS_PER_RUN = 1

OPTIONS = {
    ".checkpoint": {
        "flags": {
            "base": [],
            # Add trained checkpoints here, e.g.:
            # "zo_lora_run1": ["--checkpoint", "/path/to/run/checkpoint.pt"],
        },
        "name": "ckpt",
    },
    ".split": {
        "flags": {
            "test":          ["--split", "test"],
            "test_ood_year": ["--split", "test_ood_year"],
        },
        "name": "split",
    },
}

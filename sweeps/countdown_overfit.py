#!/usr/bin/env mlsweep_run

COMMAND = [
    "python", "train.py",
    "--model.model-id", "Qwen/Qwen3-4B",
    "--objective", "(countdown)",
    "--zo.mom.momentum-method", "stored_ema",
    "--zo.mom.second-moment",
    "--zo.base.overfit-first-batch",
    "--training.steps", "200",
    "--zo.batch-size", "16",
    "--zo.z-batch", "16",
    "--lora.strategy", "standard",
    "--lora.r", "16",
]

OPTIONS = {}

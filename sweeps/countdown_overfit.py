#!/usr/bin/env mlsweep_run

COMMAND = [
    "python", "train.py",
    "--model.model_id", "Qwen/Qwen3-4B",
    "--objective", "(countdown)",
    "--zo.mom.momentum_method", "stored_ema",
    "--zo.mom.second-moment",
    "--zo.base.overfit_first_batch",
    "--training.steps", "200",
    "--zo.batch_size", "16",
    "--zo.z_batch", "16",
    "--lora.strategy", "standard",
    "--lora.r", "16",
]

OPTIONS = {}

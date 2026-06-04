#!/usr/bin/env mlsweep_run
# Quick end-to-end smoke test: verifies the worker can run a job, imports work,
# and CUDA is available. No model loading, completes in seconds.

COMMAND = [
    "python", "-c",
    "import torch, transformers, datasets; "
    "from mlsweep.logger import MLSweepLogger; "
    "assert torch.cuda.is_available(), 'no CUDA'; "
    "print('device:', torch.cuda.get_device_name(0)); "
    "print('transformers:', transformers.__version__); "
    "logger = MLSweepLogger(); "
    "logger.log({'smoke': 1.0}); "
    "logger.close(); "
    "print('smoke ok') ",
]

GPUS_PER_RUN = 1

OPTIONS = {}  # single unnamed run → name will be "smoke_default"

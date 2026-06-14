#!/usr/bin/env mlsweep_run

COMMAND = ["bash", "run.sh", "train.py"]

GPUS_PER_RUN = 1   # each condition uses one GPU

OPTIMIZERS = ["fo", "zo"]
STRATEGIES = ["none", "standard", "relora", "continual"]

# Edit to run a subset, e.g.: OPTIMIZERS = ["zo"]
RUN_OPTIMIZERS = OPTIMIZERS
RUN_STRATEGIES = STRATEGIES

OPTIONS = {
    ".optimizer": {
        "values": RUN_OPTIMIZERS,
        "flags": "--optimizer",
        "name": "",
    },
    ".lora_strategy": {
        "values": RUN_STRATEGIES,
        "flags": "--lora.strategy",
        "name": "",
    },
}

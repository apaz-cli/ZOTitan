#!/usr/bin/env mlsweep_run
# Run a single condition. Set which one via env vars:
#
#   OPTIMIZER=fo LORA_STRATEGY=none mlsweep_run sweeps/runone.py --manager <url>
#
# Defaults to fo/none if unset.

import os

COMMAND = ["bash", "run.sh", "train.py"]

GPUS_PER_RUN = 1

_OPTIMIZERS = ["fo", "zo"]
_STRATEGIES = ["none", "standard", "relora", "continual"]

_opt      = os.environ.get("OPTIMIZER", "fo")
_strategy = os.environ.get("LORA_STRATEGY", "none")
assert _opt      in _OPTIMIZERS, f"OPTIMIZER={_opt!r} not in {_OPTIMIZERS}"
assert _strategy in _STRATEGIES, f"LORA_STRATEGY={_strategy!r} not in {_STRATEGIES}"

OPTIONS = {
    ".optimizer": {
        "values": [_opt],
        "flags": "--optimizer",
        "name": "",
    },
    ".lora_strategy": {
        "values": [_strategy],
        "flags": "--lora.strategy",
        "name": "",
    },
}

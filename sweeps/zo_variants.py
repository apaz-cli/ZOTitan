#!/usr/bin/env mlsweep_run
# Ablation over ZO optimizer variants. The two axes cross to the full first/second-moment
# matrix in MomentumConfig: MeZO, MeZO-momentum (stored/memory-free), ZO-RMSProp, ZO-Adam.
#
# Axes:
#   momentum     — none | stored_ema | seed_window   (first moment)
#   second_moment — off | on                          (Adam/RMSProp denominator)

COMMAND      = ["bash", "run.sh", "train.py", "--optimizer", "zo", "--lora.strategy", "none"]
GPUS_PER_RUN = 1

OPTIONS = {
    ".momentum": {
        "flags": {
            "none":        [],
            "stored_ema":  ["--zo.mom.momentum_method", "stored_ema"],
            "seed_window": ["--zo.mom.momentum_method", "seed_window", "--zo.seed_window.size", "20"],
        },
        "name": "mom",
    },
    ".second_moment": {
        "flags": {
            "off": [],
            "on":  ["--zo.mom.second_moment"],
        },
        "name": "2m",
    },
}

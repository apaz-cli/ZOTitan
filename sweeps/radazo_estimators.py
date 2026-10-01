#!/usr/bin/env mlsweep_run
"""Does R-AdaZO's gain carry over to other gradient estimators?

Companion to radazo_optimizers.py. Each estimator our earlier sweeps found competitive
(LOZO at rank 256, ZO-Muon/polar at rank 16, OpenAI-ES centered ranks) is run under
ZO-RMSProp (the update rule those sweeps used) and under R-AdaZO, β₂ = 0.99.

3 estimators × 2 rules × 8 lr = 48 runs.
"""

COMMAND = [
    "python", "train.py",
    "--optimizer", "zo",
    "--objective", "tinystories",
    "--model.model-id", "zotitan/tinystories-byte-rnn",
    "--model.max-seq-len", "256",
    "--training.steps", "1000",
    "--training.compile-mode", "none",
    "--zo.base.ckpt-every", "1000000",
    "--zo.base.loss-kill-threshold", "10",
    "--zo.base.loss-kill-threshold-patience", "5",
]

GPUS_PER_RUN = 1
METRIC = "ppl"
GOAL = "minimize"

EXTRA_FLAGS = ["--zo.batch-size", "16", "--zo.z-batch", "16", "--zo.eps", "0.0012",
               "--zo.mom.beta1", "0.9", "--zo.mom.beta2", "0.99", "--zo.mom.second-moment"]

OPTIONS = {
    ".est": {
        ".lozo":  {"flags": ["--zo.perturbation.distribution", "low_rank", "--zo.perturbation.rank", "256"]},
        ".polar": {"flags": ["--zo.perturbation.distribution", "polar", "--zo.perturbation.rank", "16"]},
        ".es":    {"flags": ["--zo.fitness.strategy", "centered_rank", "--zo.shared-batch"]},
    },
    ".opt": {
        "flags": {
            "rmsprop": [],
            "radazo":  ["--zo.mom.momentum-method", "stored_ema",
                        "--zo.mom.second-moment-source", "momentum", "--zo.mom.no-bias-correction"],
        },
    },
    ".lr": {
        "values": ["1e-4", "1.8e-4", "3.2e-4", "5.6e-4", "1e-3", "1.8e-3", "3.2e-3", "5.6e-3"],
        "flags": "--zo.base.lr",
        "name": "lr",
    },
}

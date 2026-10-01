#!/usr/bin/env mlsweep_run
"""R-AdaZO vs every ZO update rule we have, each over its own lr sweep.

R-AdaZO (Shu et al., ICML 2025, arXiv 2502.01014) is ZO-AdaMM with the second moment
taken over the first moment, v ← β₂v + (1-β₂)m², and no bias correction. The paper's
baselines are ZO-SGD, ZO-signSGD, ZO-AdaMM and ZO-RMSProp; we add ZO-momentum, Signum,
and our usual bias-corrected ZO-Adam. AdaMM and R-AdaZO are each run with and without
bias correction, so the m² change and the bias-correction change separate cleanly.

Adaptive rules also sweep β₂: 0.99 is the paper's setting, 0.999 ours.
Ranked on held-out tinystories ppl (1000 val examples) after 1000 steps, not min train
loss, which rewards runs that dip and then diverge.

Estimator variants (LOZO, ZO-Muon, ES) are in radazo_estimators.py.

4 rules × 8 lr + 5 adaptive rules × 2 β₂ × 8 lr = 112 runs.
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

# Known-good estimator settings (findings/FINDINGS.md); eps is a flat plateau.
EXTRA_FLAGS = ["--zo.batch-size", "16", "--zo.z-batch", "16", "--zo.eps", "0.0012",
               "--zo.mom.beta1", "0.9"]

EMA      = ["--zo.mom.momentum-method", "stored_ema"]
SECOND   = ["--zo.mom.second-moment"]
FROM_M   = ["--zo.mom.second-moment-source", "momentum"]
NO_BC    = ["--zo.mom.no-bias-correction"]

BETA2 = {"values": ["0.99", "0.999"], "flags": "--zo.mom.beta2", "name": "b2"}

OPTIONS = {
    ".opt": {
        ".mezo":      {"flags": []},                                       # ZO-SGD
        ".mezomom":   {"flags": EMA},                                      # ZO-SGD + momentum
        ".signsgd":   {"flags": ["--zo.mom.sign-update"]},                 # ZO-signSGD
        ".signum":    {"flags": EMA + ["--zo.mom.sign-update"]},           # ZO-Signum
        ".rmsprop":   {"flags": SECOND, ".b2": BETA2},                     # ZO-RMSProp
        ".adam":      {"flags": EMA + SECOND, ".b2": BETA2},               # ZO-Adam (bias-corrected)
        ".adamm":     {"flags": EMA + SECOND + NO_BC, ".b2": BETA2},       # ZO-AdaMM as in the paper
        ".radazo":    {"flags": EMA + SECOND + FROM_M + NO_BC, ".b2": BETA2},  # R-AdaZO (Algorithm 2)
        ".radazobc":  {"flags": EMA + SECOND + FROM_M, ".b2": BETA2},      # R-AdaZO + bias correction
    },
    # Quarter-decade grid. Prior sweeps on this model: optimum ~2.5e-3 for every rule,
    # divergence by 1e-2, so 1e-4..5.6e-3 brackets the cliff on both sides.
    ".lr": {
        "values": ["1e-4", "1.8e-4", "3.2e-4", "5.6e-4", "1e-3", "1.8e-3", "3.2e-3", "5.6e-3"],
        "flags": "--zo.base.lr",
        "name": "lr",
    },
}

#!/usr/bin/env mlsweep_run
"""lr x clipping ablation: can clipping let us push lr past the divergence cliff?

At hidden=1024 (~10M body), lr 2.5e-3 is the stable optimum and lr >= 3e-3
diverges. Here we sweep lr into the diverging regime against the ZO proj-grad
clipping strategies (adaptive quantile threshold tau):

    none:   no clipping (baseline)
    pp10:   per_pair clamp to +-tau, clip_pct=0.1 (top 10% clipped)
    pp20:   per_pair clamp to +-tau, clip_pct=0.2 (more aggressive)
    norm10: scale the whole grad so RMS <= tau, clip_pct=0.1

lr = [2.5e-3, 3e-3, 4e-3, 6e-3]. eps = 0.0012, bs16_z16 fixed.

4 lrs x 4 clip settings = 16 runs.
"""

COMMAND = [
    "python", "train.py",
    "--optimizer", "zo",
    "--objective", "tinystories",
    "--model.model-id", "zotitan/tinystories-byte-rnn",
    "--model.max-seq-len", "256",
    "--training.steps", "1000",
    "--training.no-eval",
    "--training.compile-mode", "none",
    "--zo.mom.momentum-method", "stored_ema",
    "--zo.mom.second-moment",
    "--zo.base.ckpt-every", "1000000",
    "--zo.base.loss-kill-threshold", "10",
    "--zo.base.loss-kill-threshold-patience", "5",
]

GPUS_PER_RUN = 1

# Held fixed: bs16_z16 and eps = 0.0012.
EXTRA_FLAGS = ["--zo.batch-size", "16", "--zo.z-batch", "16", "--zo.eps", "0.0012"]

OPTIONS = {
    ".lr": {
        "values": [0.0025, 0.003, 0.004, 0.006],
        "flags": "--zo.base.lr",
        "name": "lr",
    },
    ".clip": {
        "flags": {
            "none":   [],
            "pp10":   ["--zo.clip.strategy", "per_pair", "--zo.clip.clip-pct", "0.1"],
            "pp20":   ["--zo.clip.strategy", "per_pair", "--zo.clip.clip-pct", "0.2"],
            "norm10": ["--zo.clip.strategy", "norm", "--zo.clip.clip-pct", "0.1"],
        },
        "name": "clip",
    },
}

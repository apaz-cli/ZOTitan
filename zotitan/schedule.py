import math
import torch
from dataclasses import dataclass, field


@dataclass
class WSDConfig:
    warmup_frac: float = 0.0
    """Fraction of total_steps for linear warmup (lo → hi)."""

    decay_frac: float = 0.0
    """Fraction of total_steps for cosine decay (hi → lo).
    Stable fraction = 1 - warmup_frac - decay_frac."""


@dataclass
class BaseTrainConfig:
    lr: float = 2e-5
    """Peak learning rate."""

    lr_wsd: WSDConfig = field(default_factory=WSDConfig)
    """WSD schedule for the learning rate. Default (decay_frac=0) = constant at lr."""

    grad_clip: float = 1.0
    """Gradient clipping threshold."""

    ckpt_every: int = 2_000
    """Save a checkpoint every N steps."""

    overfit_first_batch: bool = False
    """Debug: draw one batch up front and reuse it every step (loss should drive toward 0).
    A quick way to smoke out objective/optimizer bugs in isolation from the data pipeline."""


def maybe_torchcompile(fn=None, *, enabled: bool = True, mode: str | None = None):
    if fn is None:
        return lambda f: maybe_torchcompile(f, enabled=enabled, mode=mode)
    if not enabled:
        return fn
    kwargs: dict = {}
    if mode is not None:
        kwargs["mode"] = mode
    return torch.compile(fn, **kwargs)


def wsd_is_constant(cfg: WSDConfig) -> bool:
    """True when the schedule never changes the LR (no warmup, no decay)."""
    return cfg.warmup_frac == 0.0 and cfg.decay_frac == 0.0


def wsd_value(step: int, total_steps: int, lo: float, hi: float, cfg: WSDConfig) -> float:
    """Evaluate a Warmup-Stable-Decay schedule at the given step.
    Returns hi immediately when total_steps is 0 (no schedule)."""
    if total_steps == 0:
        return hi
    warmup_end  = cfg.warmup_frac * total_steps
    decay_start = (1.0 - cfg.decay_frac) * total_steps
    if step < warmup_end:
        return lo + (hi - lo) * step / max(warmup_end, 1.0)
    if step < decay_start:
        return hi
    t = min((step - decay_start) / max(total_steps - decay_start, 1.0), 1.0)
    return lo + 0.5 * (hi - lo) * (1.0 + math.cos(math.pi * t))

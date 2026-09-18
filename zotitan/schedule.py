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

    loss_kill_threshold: float = 0.0
    """Early-termination threshold on the per-step loss. If `loss` stays above this
    value for `loss_kill_threshold_patience` consecutive steps, training stops
    gracefully. 0 disables."""

    loss_kill_threshold_patience: int = 0
    """Consecutive steps `loss` must exceed `loss_kill_threshold` before training
    stops. Only active when both are > 0."""


class LossKillGuard:
    """Per-step early-termination check shared by the FO and ZO loops.

    Stops once `loss` has stayed above `loss_kill_threshold` for
    `loss_kill_threshold_patience` consecutive steps; inactive (never stops) when
    either knob is 0. Copies the two thresholds out of the config rather than
    holding it, so the guard outlives nothing it doesn't need."""

    def __init__(self, cfg: BaseTrainConfig):
        self.threshold = cfg.loss_kill_threshold
        self.patience  = cfg.loss_kill_threshold_patience
        self.active    = self.threshold > 0.0 and self.patience > 0
        self.bad_steps = 0

    def should_stop(self, loss: float) -> bool:
        if not self.active:
            return False
        self.bad_steps = self.bad_steps + 1 if loss > self.threshold else 0
        if self.bad_steps < self.patience:
            return False
        print(f"  loss exceeded {self.threshold} for {self.bad_steps} consecutive steps — stopping early.")
        return True


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

import os
import time
import torch
from dataclasses import dataclass, field
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from .lora import save_merged_checkpoint
from .objective import floatify
from .pretty import print_config, print_fo_step
from . import profiling
from .profiling import ProfilingConfig, maybe_enable_profiling
from .schedule import WSDConfig, wsd_value, wsd_is_constant, BaseTrainConfig


@dataclass
class FOConfig:
    base: BaseTrainConfig = field(default_factory=BaseTrainConfig)

    batch_size: int = 40
    """Per-step batch size."""

    weight_decay: float = 0.0
    """Weight decay coefficient, passed directly to AdamW. 0 = disabled."""


def train_fo(model, tokenizer, total_steps, seed, merge_fn, logger, cfg: FOConfig | None = None, objective=None, profiling_cfg: ProfilingConfig | None = None):
    """
    Standard first-order training loop.
    Trains only parameters where requires_grad=True (i.e. LoRA adapters).
    If merge_fn is provided, calls merge_fn(model) after each optimizer step.
    """
    if cfg is None:
        cfg = FOConfig()
    if objective is None:
        from objective import make_objective
        objective = make_objective("scijudge")

    base     = cfg.base
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=base.lr,
                      weight_decay=cfg.weight_decay)
    loader    = objective.train_batches(tokenizer, seed, cfg.batch_size)
    device    = next(model.parameters()).device
    run_dir   = os.environ.get("MLSWEEP_RUN_DIR", ".")
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print_config(cfg)
    print(f"  FO train: {total_steps:,} steps | {n_params:,} params")
    model.train()
    show_lr    = not wsd_is_constant(base.lr_wsd)   # a constant LR is already in the config dump
    overfit    = base.overfit_first_batch
    if overfit:
        print("  overfit: reusing the first batch every step")
    fixed_batch = None                             # the cached first batch (overfit only)

    with maybe_enable_profiling(profiling_cfg or ProfilingConfig(), run_dir=run_dir) as torch_profiler:
        for step in range(total_steps):
            if fixed_batch is not None:
                batch = fixed_batch
            else:
                batch = objective.to_device(next(loader), device)
                if overfit:
                    fixed_batch = batch

            lr = wsd_value(step, total_steps, lo=0.0, hi=base.lr, cfg=base.lr_wsd)
            for g in optimizer.param_groups:
                g["lr"] = lr

            t0 = time.perf_counter()
            optimizer.zero_grad()
            score = objective.score(model, batch)        # value = scalar to minimize
            value = score.value
            assert isinstance(value, torch.Tensor), "FO requires a differentiable (tensor) objective value"
            value.backward()
            grad_norm = clip_grad_norm_(model.parameters(), base.grad_clip).item()
            optimizer.step()

            if merge_fn is not None:
                merge_fn(model)

            sm = floatify(score.metrics)                 # ce, z_loss, acc, … (per criterion)
            loss = value.item()
            step_time = time.perf_counter() - t0
            metrics = {"loss": loss, "grad_norm": grad_norm, "lr": lr, "step_time": step_time, **sm}
            logger.log(metrics)
            print_fo_step(step, total_steps, loss, grad_norm, lr, extra=sm or None,
                          show_lr=show_lr, step_time=step_time)

            if torch_profiler:
                torch_profiler.step()
                if profiling.trace_saved:
                    break

            if (step + 1) % base.ckpt_every == 0:
                save_merged_checkpoint(model, tokenizer, run_dir)
                logger.sync()

    return model

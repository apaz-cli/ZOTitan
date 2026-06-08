#!/usr/bin/env python

# mlsweep worker sets CUDA_VISIBLE_DEVICES, MLSWEEP_RUN_DIR, MLSWEEP_WORKER_SOCKET
import tyro
import torch
from dataclasses import dataclass, field
from typing import Literal
from mlsweep.logger import MLSweepLogger

from .model import load_model, ModelConfig
from .lora  import make_lora, merge_and_unload, merge_and_zero_lora, print_trainable_params, LoRAConfig
from .train_fo import train_fo, FOConfig
from .train_zo import train_zo, ZOConfig
from .objective import build_objective, parse_objective_spec, prepare_datasets
from .profiling import ProfilingConfig
from .pretty import print_eval_results


@dataclass
class TrainingConfig:
    steps: int = 20_000
    """Total training steps."""

    eval: bool = True
    """Run evaluation after training. Set to False to skip (useful for smoke tests)."""

    compile_mode: Literal["none", "default", "max-autotune", "max_autotune"] = "default"
    """torch.compile() mode.
    max-autotune and max_autotune are identical."""

    compile_cudagraphs: bool = True
    """Enable cudagraphs. Works with your compile_mode.
    """


@dataclass
class ExperimentConfig:
    optimizer: Literal["fo", "zo"] = "zo"
    """Which optimizer to use: first-order (AdamW) or zeroth-order (MeZO)."""

    model: ModelConfig = field(default_factory=ModelConfig)
    """Model ID and init mode (pretrained vs random)."""

    objective: str = "scijudge"
    """Objective spec: a '+'-joined mixture of NAME or (NAME[,loss_weight[,data_weight]]).
    e.g. "scijudge", "c4", "(c4,.5)+scijudge". NAME selects each component's dataset; the
    spec sets per-component (normalized) mix weights and any per-objective kwargs (e.g.
    loss settings). See the README for the full grammar."""

    training: TrainingConfig = field(default_factory=TrainingConfig)
    """Step counts and cycle schedule."""

    lora: LoRAConfig = field(default_factory=LoRAConfig)
    """LoRA adapter configuration (rank, alpha, target modules)."""

    fo: FOConfig = field(default_factory=FOConfig)
    """First-order optimizer configuration (active when optimizer=fo)."""

    zo: ZOConfig = field(default_factory=ZOConfig)
    """Zeroth-order optimizer configuration (active when optimizer=zo)."""

    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    """PyTorch profiler (chrome trace / Perfetto)."""

def _compile_args(compile_mode: str, cudagraphs: bool) -> tuple[bool, str | None]:
    """Map user-facing knobs to (enabled, torch_compile_mode)."""
    compile_mode = compile_mode.replace("_", "-")
    if compile_mode == "none":
        return False, None
    if compile_mode == "default":
        return True, "reduce-overhead" if cudagraphs else "default"
    if compile_mode == "max-autotune":
        return True, "max-autotune" if cudagraphs else "max-autotune-no-cudagraphs"
    raise AssertionError(f"unreachable compile_mode: {compile_mode!r}")

def _run(cfg: ExperimentConfig, logger: MLSweepLogger):
    tc            = cfg.training
    lora_cfg      = cfg.lora
    compile_enabled, compile_mode = _compile_args(tc.compile_mode, tc.compile_cudagraphs)
    objs          = parse_objective_spec(cfg.objective)
    objective     = build_objective(objs, compile_enabled=compile_enabled,
                                    compile_mode=compile_mode, max_seq_len=cfg.model.max_seq_len)
    prepare_datasets(objective)
    optimizer     = cfg.optimizer
    lora_strategy = cfg.lora.strategy
    label         = f"{optimizer}/{lora_strategy}"

    if optimizer == "fo" and not objective.differentiable:
        raise ValueError(f"FO requires a differentiable objective; {objective.name} is not.")

    def _train(model, tokenizer, steps, seed=0, merge_fn=None):            
        if optimizer == "fo":
            return train_fo(model, tokenizer, steps, seed=seed, merge_fn=merge_fn,
                            logger=logger, cfg=cfg.fo, objective=objective,
                            profiling_cfg=cfg.profiling)
        else:
            return train_zo(model, tokenizer, steps, seed=seed, merge_fn=merge_fn,
                            logger=logger, cfg=cfg.zo, objective=objective,
                            profiling_cfg=cfg.profiling)

    model, tokenizer = load_model(cfg.model)

    if lora_strategy == "none":
        print(f"[{label}] {tc.steps:,} steps (full finetune)")
        print_trainable_params(model)
        model = _train(model, tokenizer, tc.steps)

    elif lora_strategy == "standard":
        print(f"[{label}] {tc.steps:,} steps")
        model = make_lora(model, lora_cfg)
        model = _train(model, tokenizer, tc.steps)
        model = merge_and_unload(model)

    elif lora_strategy == "relora":
        if lora_cfg.relora_cycles < 1:
            raise ValueError("lora.strategy=relora requires lora.relora-cycles >= 1")
        steps_per = tc.steps // lora_cfg.relora_cycles
        print(f"[{label}] {lora_cfg.relora_cycles} × {steps_per:,} steps")
        for i in range(lora_cfg.relora_cycles):
            print(f"    cycle {i+1}/{lora_cfg.relora_cycles}")
            model = make_lora(model, lora_cfg)
            _train(model, tokenizer, steps_per, seed=i)
            model = merge_and_unload(model)

    elif lora_strategy == "continual":
        print(f"[{label}] {tc.steps:,} steps (merge every step)")
        model = make_lora(model, lora_cfg)
        model = _train(model, tokenizer, tc.steps, merge_fn=merge_and_zero_lora)
        model = merge_and_unload(model)

    else:
        raise AssertionError(f"unreachable lora_strategy: {lora_strategy!r}")

    if tc.eval:
        results = objective.evaluate(model, tokenizer)
        logger.log(results)
        print_eval_results(results)

def run(cfg: ExperimentConfig, logger: MLSweepLogger):
    try:
        return _run(cfg, logger)
    except KeyboardInterrupt:
        print("\nInterrupted by user, exiting early.")

def main():
    cfg = tyro.cli(ExperimentConfig)
    with MLSweepLogger() as logger:
        run(cfg, logger)


if __name__ == "__main__":
    main()

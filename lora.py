import math
import os
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Literal

DTypeStr = Literal["bf16", "fp32"]

LORA_R       = 16
LORA_ALPHA   = 32
LORA_TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"}

_LORA_DTYPE_MAP: dict[DTypeStr, torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


@dataclass
class LoRAConfig:
    strategy: Literal["none", "standard", "relora", "continual"] = "none"
    """LoRA training strategy.
    none:      no LoRA — full-parameter finetuning (train the base weights directly).
    standard:  standard LoRA — train adapters for all steps, merge at the end.
    relora:    cyclic merge-and-reinit (ReLoRA).
    continual: merge into base weights every step, keeping adapter wrapper alive."""

    relora_cycles: int = 5
    """Number of ReLoRA merge-and-reinit cycles (only used when strategy=relora)."""

    r: int = LORA_R
    """Adapter rank."""

    alpha: int = LORA_ALPHA
    """Scaling factor; effective scale = alpha / r."""

    targets: list[str] = field(default_factory=lambda: list(LORA_TARGETS))
    """Linear module name suffixes to wrap with LoRA."""

    dtype: DTypeStr = "fp32"
    """Dtype for LoRA A and B matrices. fp32 (default) is useful for ZO, because
    perturbations are often small and you don't want them rounded away. Using
    bf16 halves the trainable-parameter memory, but increases sensitivity to your
    choice of ZO epsilon."""

    init: Literal["kaiming", "gaussian"] = "kaiming"
    """Initialisation for lora_A. lora_B is always zero-init.
    kaiming: kaiming_uniform_ with a=√5 (standard LoRA).
    gaussian: kaiming_normal_ (same fan-in variance, normal distribution)."""


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, lora_alpha: int, dtype: DTypeStr = "fp32", init: str = "kaiming"):
        super().__init__()
        self.base_linear = base
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)

        # Trainable adapters are kept in the configured dtype (fp32 default).
        lora_dtype = _LORA_DTYPE_MAP[dtype]
        device = base.weight.device
        self.lora_A = nn.Linear(base.in_features, r, bias=False, dtype=lora_dtype, device=device)
        self.lora_B = nn.Linear(r, base.out_features, bias=False, dtype=lora_dtype, device=device)
        self.lora_dtype = lora_dtype

        if init == "kaiming":
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        else:
            nn.init.kaiming_normal_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)
        self.scaling = lora_alpha / r

    def forward(self, x):
        # Compute the base forward
        base_out = self.base_linear(x)

        # Run the adapter matmul in the lora dtype, to do this we must cast the input.
        # The adapter weights are already in the lora dtype, so no cast there.
        x = x.to(self.lora_dtype)
        delta = self.lora_B(self.lora_A(x)) * self.scaling

        # And append the lora delta
        return base_out + delta.to(base_out.dtype)


def make_lora(model, cfg: LoRAConfig | None = None):
    if cfg is None:
        cfg = LoRAConfig()
    targets = set(cfg.targets)
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(t) for t in targets):
            continue
        *parent_parts, child_name = name.split(".")
        parent = model
        for part in parent_parts:
            parent = getattr(parent, part)
        setattr(parent, child_name, LoRALinear(module, cfg.r, cfg.alpha, cfg.dtype, cfg.init))
    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            param.requires_grad_(False)
    print_trainable_params(model)
    return model


def print_trainable_params(model):
    """Log how many parameters are trainable (requires_grad) vs total."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


def merge_and_unload(model):
    """Fold each LoRALinear delta into its base weight and replace with plain nn.Linear."""
    print("  merging LoRA into base weights and unloading adapters...")
    for name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue
        *parent_parts, child_name = name.split(".")
        parent = model
        for part in parent_parts:
            parent = getattr(parent, part)
        module.base_linear.weight.data.add_(module.lora_B.weight @ module.lora_A.weight, alpha=module.scaling)
        setattr(parent, child_name, module.base_linear)
    return model


def merge_and_zero_lora(model):
    """
    Fold each LoRALinear delta into base weight, zero lora_B. Keeps wrapper and
    optimizer alive — no structural change, so compiled graph stays valid.

    For FO-Steps: Adam's m/v on lora_B survive across calls. Each step starts with
    lora_B=0 but inherits accumulated momentum, so the optimizer continues tracking
    movement in the fixed lora_A basis rather than resetting direction.

    For ZO-Steps: no optimizer state exists; this is equivalent to standard ZO
    parameterized through a per-step low-rank path.
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.base_linear.weight.data.add_(module.lora_B.weight @ module.lora_A.weight, alpha=module.scaling)
            module.lora_B.weight.data.zero_()


def merged_state_dict(model):
    """Full CPU state dict with LoRA deltas folded into the base weights, and keys
    renamed to plain HF names (``q_proj.weight`` etc.). Does not disturb the live
    model or optimizer.
    """
    out = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            delta = (module.lora_B.weight @ module.lora_A.weight).float() * module.scaling
            weight = (module.base_linear.weight.float() + delta).to(module.base_linear.weight.dtype)
            out[name + ".weight"] = weight.detach().cpu()
            if module.base_linear.bias is not None:
                out[name + ".bias"] = module.base_linear.bias.detach().cpu()
    # Materialize on CPU
    for key, value in model.state_dict().items():
        if ".base_linear." in key or ".lora_A." in key or ".lora_B." in key:
            continue
        out[key] = value.detach().cpu()
    return out


def save_merged_checkpoint(model, tokenizer, run_dir):
    """Save a merged HF checkpoint (LoRA deltas folded into the base weights) plus
    the tokenizer under ``<run_dir>/checkpoint``."""
    ckpt_dir = os.path.join(run_dir, "checkpoint")
    model.save_pretrained(ckpt_dir, state_dict=merged_state_dict(model))
    tokenizer.save_pretrained(ckpt_dir)

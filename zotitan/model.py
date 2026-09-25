import torch
from dataclasses import dataclass
from typing import Literal

DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B-Base"


@dataclass
class ModelConfig:
    model_id: str = DEFAULT_MODEL_ID
    """HuggingFace model ID."""

    pretrained: bool = True
    """True: load pretrained weights (finetuning). False: random init (pretraining from scratch)."""

    attn_impl: Literal["eager", "sdpa", "flash_attention_2"] = "sdpa"
    """Attention backend passed to from_pretrained.
    """

    max_seq_len: int = 1792
    """Max sequence length the objectives truncate tokenized inputs to. A property of the
    model's usable context; the single source of truth, threaded into each objective at
    build time (objectives never reach for a global)."""

    hidden: int | None = None
    """Override the GRU hidden size for tiny models (width scaling). None = arch default."""

    num_layers: int | None = None
    """Override the GRU layer count for tiny models (depth scaling). None = arch default."""

    num_experts: int | None = None
    """Number of MoE experts for tiny models. None = arch default (1 = no MoE)."""

    top_k: int | None = None
    """Top-k experts routed to per token for tiny models. None = arch default."""


def load_model(cfg: ModelConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    # Local (non-HF) toy models built in pure torch: selecting one as model_id builds
    # it from tiny_model.py instead of downloading pretrained weights. TINY_ARCHS is
    # the only list of which ids exist — don't mirror it here, or the two drift.
    from .tiny_model import TINY_ARCHS, build_tiny
    if cfg.model_id in TINY_ARCHS:
        model, tokenizer = build_tiny(cfg.model_id, hidden=cfg.hidden, num_layers=cfg.num_layers,
                                      num_experts=cfg.num_experts, top_k=cfg.top_k)
        return model.to(dtype=torch.bfloat16).cuda(), tokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    assert tokenizer.pad_token_id is not None, \
        "tokenizer has no pad_token — add: tokenizer.pad_token = tokenizer.eos_token"
    if cfg.pretrained:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation=cfg.attn_impl,
        )
    else:
        config = AutoConfig.from_pretrained(cfg.model_id)
        model = AutoModelForCausalLM.from_config(config)
        model = model.to(dtype=torch.bfloat16).cuda()
    return model, tokenizer

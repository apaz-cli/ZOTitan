import torch
from dataclasses import dataclass
from typing import Literal

DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B-Base"


@dataclass
class ModelConfig:
    model_id: str = DEFAULT_MODEL_ID
    """HuggingFace model ID, or a local arch spec such as `gru` or
    `(transformer,d_model=512,n_layers=8)` (see arch.py; random init, pretrained is ignored)."""

    pretrained: bool = True
    """True: load pretrained weights (finetuning). False: random init (pretraining from scratch)."""

    attn_impl: Literal["eager", "sdpa", "flash_attention_2"] = "sdpa"
    """Attention backend passed to from_pretrained.
    """

    max_seq_len: int = 1792
    """Max sequence length the objectives truncate tokenized inputs to. A property of the
    model's usable context; the single source of truth, threaded into each objective at
    build time (objectives never reach for a global)."""


def load_model(cfg: ModelConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    # Local pure-torch archs (arch.py's registry) build from scratch instead of
    # downloading. ARCHS is the only list of which ids exist — don't mirror it here.
    from .arch import build_arch, is_arch_spec
    if is_arch_spec(cfg.model_id):
        model, tokenizer = build_arch(cfg.model_id)
        return model.to(dtype=torch.bfloat16).cuda(), tokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    assert tokenizer.pad_token_id is not None, \
        "tokenizer has no pad_token — add: tokenizer.pad_token = tokenizer.eos_token"
    if cfg.pretrained:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            dtype=torch.bfloat16,
            device_map=f"cuda:{torch.cuda.current_device()}",   # plain "cuda" means cuda:0
            attn_implementation=cfg.attn_impl,
        )
    else:
        config = AutoConfig.from_pretrained(cfg.model_id)
        model = AutoModelForCausalLM.from_config(config)
        model = model.to(dtype=torch.bfloat16).cuda()
    return model, tokenizer

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


def load_model(cfg: ModelConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
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

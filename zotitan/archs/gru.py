"""Stacked GRU LM: the cheap non-transformer baseline the tinystories findings use."""
import torch
import torch.nn as nn

from ..arch import ZOLM, register_arch, register_preset


@register_arch("gru")
class GRULM(ZOLM):
    def __init__(self, vocab_size: int, pad_token_id: int, d_model: int = 256,
                 hidden: int = 1024, num_layers: int = 2):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.rnn     = nn.GRU(d_model, hidden, num_layers=num_layers, batch_first=True)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)

    def body(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        # attention_mask is ignored: the pad token's embedding is zeroed via padding_idx,
        # so padded positions contribute nothing (only a little recurrent bias drift),
        # and the CE loss masks them.
        out, _ = self.rnn(self.embed(input_ids))
        return out


# The ids the existing tinystories sweeps and findings use.
register_preset("zotitan/tinystories-byte-rnn", "gru", tokenizer="byte", d_model=256)
register_preset("zotitan/tinystories-rnn",      "gru", tokenizer="qwen", d_model=128)

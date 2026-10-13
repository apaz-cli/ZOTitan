"""Reference decoder-only transformer: the baseline new architectures are measured against.

Standard modern recipe: pre-norm RMSNorm, RoPE, causal SDPA, SwiGLU MLP, untied head.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..arch import ZOLM, register_arch


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate-half RoPE on (B, nh, T, hd)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.n_heads   = n_heads
        self.attn_norm = nn.RMSNorm(d_model)
        self.qkv       = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_out  = nn.Linear(d_model, d_model, bias=False)
        self.mlp_norm  = nn.RMSNorm(d_model)
        self.gate_up   = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.mlp_out   = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = (self.qkv(self.attn_norm(x))
                   .view(B, T, 3, self.n_heads, D // self.n_heads)
                   .permute(2, 0, 3, 1, 4))                      # 3 x (B, nh, T, hd)
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        # Right padding + causal masking: real tokens never attend to the pads after them.
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.attn_out(a.transpose(1, 2).reshape(B, T, D))
        gate, up = self.gate_up(self.mlp_norm(x)).chunk(2, dim=-1)
        return x + self.mlp_out(F.silu(gate) * up)


@register_arch("transformer")
class TransformerLM(ZOLM):
    def __init__(self, vocab_size: int, pad_token_id: int, d_model: int = 256,
                 n_layers: int = 4, n_heads: int = 4, d_ff: int | None = None,
                 max_seq_len: int = 4096, rope_base: float = 10_000.0):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} not divisible by n_heads={n_heads}"
        head_dim = d_model // n_heads
        assert head_dim % 2 == 0, f"head_dim={head_dim} must be even for RoPE"
        d_ff = d_ff if d_ff is not None else 4 * d_model * 2 // 3   # SwiGLU at ~4x MLP params

        self.embed   = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.blocks  = nn.ModuleList(Block(d_model, n_heads, d_ff) for _ in range(n_layers))
        self.norm    = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        inv_freq = rope_base ** (-torch.arange(0, head_dim, 2).float() / head_dim)
        ang = torch.outer(torch.arange(max_seq_len).float(), inv_freq)   # (T_max, hd/2)
        self.register_buffer("rope_cos", ang.cos(), persistent=False)
        self.register_buffer("rope_sin", ang.sin(), persistent=False)

        # GPT-2 style init; residual output projections scaled by depth.
        for name, p in self.named_parameters():
            if p.ndim == 2:
                std = 0.02 / math.sqrt(2 * n_layers) if name.endswith(("attn_out.weight", "mlp_out.weight")) else 0.02
                nn.init.normal_(p, std=std)
        with torch.no_grad():
            self.embed.weight[pad_token_id].zero_()

    def body(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        T = input_ids.shape[1]
        cos = self.rope_cos[:T].to(self.embed.weight.dtype)
        sin = self.rope_sin[:T].to(self.embed.weight.dtype)
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)

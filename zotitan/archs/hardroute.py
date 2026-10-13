"""A deliberately non-differentiable toy LM: the template for archs backprop can't train.

Each layer routes every token to one of `n_experts` linear maps by a hard (optionally
Gumbel-noised) argmax — no gradient reaches the router — then mixes the sequence with
a causal running mean. The Gumbel noise is drawn from the default torch generator in
every forward (train or eval mode), so it also exercises the ZO loop's shared ±
randomness: both passes of a pair see the same routing noise.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..arch import ZOLM, register_arch


class HardRouteBlock(nn.Module):
    def __init__(self, d_model: int, n_experts: int, noise: float):
        super().__init__()
        self.noise   = noise
        self.norm    = nn.RMSNorm(d_model)
        self.router  = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.Parameter(torch.randn(n_experts, d_model, d_model) * d_model ** -0.5)
        self.mix     = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        logits = self.router(h)
        if self.noise > 0:
            u = torch.rand_like(logits).clamp_(1e-6, 1 - 1e-6)
            logits = logits - self.noise * torch.log(-torch.log(u))     # + Gumbel noise
        # Run every expert, keep the chosen one: (B, T, E, d) beats gathering (B, T, d, d) weights.
        y   = torch.einsum("btd,nde->btne", h, self.experts)
        idx = logits.argmax(-1)[..., None, None].expand(-1, -1, 1, y.shape[-1])
        x   = x + F.gelu(y.gather(2, idx).squeeze(2))
        # Causal running mean: the only cross-token mixing.
        t = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        return x + self.mix(x.cumsum(dim=1) / t)


@register_arch("hardroute")
class HardRouteLM(ZOLM):
    def __init__(self, vocab_size: int, pad_token_id: int, d_model: int = 128,
                 n_layers: int = 2, n_experts: int = 4, noise: float = 1.0):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.blocks  = nn.ModuleList(HardRouteBlock(d_model, n_experts, noise) for _ in range(n_layers))
        self.norm    = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def body(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

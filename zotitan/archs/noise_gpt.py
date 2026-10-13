"""The fixed model of the Noise speedrun: the 37.7M-parameter GPT of the Dust paper.

Ported from github.com/qlabs-eng/noise `utils/model.py` (MIT, (c) 2026 qlabs-eng). Same
function and the same state-dict keys and shapes, so a ZOTitan checkpoint loads strictly
into their `utils/eval.py`. Pair it with the `noise` objective (`tokenizer='noise'`).

8 blocks, width 512, 8 heads, sliding-window attention (pattern SSSL: S = half-sequence
window, L = full; the last block is always L), value embeddings on alternate blocks,
learned residual/x0 mixing scalars, relu² MLP, QK-norm, logit softcap 15.
"""
from functools import cache

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from ..arch import ZOLM, register_arch

flex = torch.compile(flex_attention)


@cache
def window_mask(window, length, device):
    return create_block_mask(
        lambda b, h, q, k: (q >= k) & (q - k <= window), None, None, length, length, device=device,
    )


def attention(q, k, v, window):
    """Causal attention with a left window; tensors are [batch, token, head, dim]. A query at
    position s sees keys at s - window to s."""
    mask = window_mask(window, q.shape[1], q.device)
    return flex(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=mask).transpose(1, 2)


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Value embeddings on every other block, counting back from the last."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * -sin + x2 * cos], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head, layer_idx, n_layer):
        super().__init__()
        self.n_head, self.n_kv_head = n_head, n_kv_head
        self.head_dim = n_embd // n_head
        assert n_embd % n_head == 0
        self.c_q = nn.Linear(n_embd, n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        # The value embedding is scaled per head by 2·sigmoid(gate), read from the block
        # input's first 32 channels.
        self.ve_gate_channels = 32
        self.ve_gate = (nn.Linear(self.ve_gate_channels, n_kv_head, bias=False)
                        if has_ve(layer_idx, n_layer) else None)

    def forward(self, x, ve, cos_sin, window):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        y = attention(q, k, v, window)
        return self.c_proj(y.contiguous().view(B, T, -1))


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head, layer_idx, n_layer):
        super().__init__()
        self.attn = CausalSelfAttention(n_embd, n_head, n_kv_head, layer_idx, n_layer)
        self.mlp = MLP(n_embd)

    def forward(self, x, ve, cos_sin, window):
        x = x + self.attn(norm(x), ve, cos_sin, window)
        return x + self.mlp(norm(x))


@register_arch("noise_gpt", tokenizer="noise")
class NoiseGPT(ZOLM):
    logit_softcap = 15.0

    def __init__(self, vocab_size: int, pad_token_id: int, sequence_len: int = 2048,
                 n_layer: int = 8, n_head: int = 8, n_kv_head: int = 8, n_embd: int = 512,
                 window_pattern: str = "SSSL", pad_vocab_size_to: int = 64,
                 writer_init: str = "uniform"):
        """`writer_init` picks the init of the residual writers (attn.c_proj, mlp.c_proj):
        'zero' is the Dust/Noise `init_weights`; 'uniform' is their EGGROLL baseline's
        U(±1.5·√(3/fan_in)), the forward-only reference. Init is free under their rules."""
        super().__init__()
        assert writer_init in ("zero", "uniform"), writer_init
        self.vocab_size = vocab_size
        padded_vocab = (vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to * pad_vocab_size_to
        # The fused loss path reads lm_head.weight whole, so padding columns would leak into it.
        assert padded_vocab == vocab_size, "padded vocab is not supported; pick pad_vocab_size_to | vocab"

        long_w, short_w = sequence_len, sequence_len // 2
        sizes = [{"L": long_w, "S": short_w}[window_pattern.upper()[i % len(window_pattern)]]
                 for i in range(n_layer)]
        sizes[-1] = long_w
        self.window_sizes = sizes

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab, n_embd),
            "h": nn.ModuleList([Block(n_embd, n_head, n_kv_head, i, n_layer) for i in range(n_layer)]),
        })
        self.lm_head = nn.Linear(n_embd, padded_vocab, bias=False)   # untied from the embedding
        # The stream entering block i is resid_lambdas[i]·x + x0_lambdas[i]·x0, a learned mix
        # of the previous block's output and the normalized token embedding.
        self.resid_lambdas = nn.Parameter(torch.ones(n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(n_layer))
        head_dim = n_embd // n_head
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(padded_vocab, n_kv_head * head_dim)
            for i in range(n_layer) if has_ve(i, n_layer)
        })
        inv_freq = 1.0 / 10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        freqs = torch.outer(torch.arange(sequence_len * 10, dtype=torch.float32), inv_freq)
        self.register_buffer("cos", freqs.cos().bfloat16()[None, :, None, :], persistent=False)
        self.register_buffer("sin", freqs.sin().bfloat16()[None, :, None, :], persistent=False)
        self._init_weights(n_embd, writer_init)

    @property
    def embed(self):
        return self.transformer.wte

    @torch.no_grad()
    def _init_weights(self, n_embd, writer_init):
        nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = 3 ** 0.5 * n_embd ** -0.5
        for block in self.transformer.h:
            for lin in (block.attn.c_q, block.attn.c_k, block.attn.c_v, block.mlp.c_fc):
                nn.init.uniform_(lin.weight, -s, s)
            for lin in (block.attn.c_proj, block.mlp.c_proj):
                if writer_init == "zero":
                    nn.init.zeros_(lin.weight)
                else:
                    w = 1.5 * (3 / lin.weight.shape[1]) ** 0.5
                    nn.init.uniform_(lin.weight, -w, w)
            if block.attn.ve_gate is not None:
                nn.init.zeros_(block.attn.ve_gate.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for ve in self.value_embeds.values():
            nn.init.uniform_(ve.weight, -s, s)

    def body(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        # attention_mask is ignored: the data is fixed-length rows with no padding.
        T = input_ids.shape[1]
        cos_sin = (self.cos[:, :T], self.sin[:, :T])
        x = x0 = norm(self.transformer.wte(input_ids))
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](input_ids) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        return norm(x)

"""Small non-transformer language models for cheap ZO experiments.

Implements just enough of the HF `PreTrainedModel` surface that ZOTitan's
train/eval loops consume, and nothing else:

  - the reference loss path (losses._torch_xentropy) calls `model(**batch)` and
    reads `.loss` / `.logits`;
  - the fused Liger path (losses._liger_xentropy) calls `get_decoder()` and
    `get_output_embeddings()` and reads `.last_hidden_state` / `head.weight`;
  - the checkpoint path calls `model.save_pretrained(dir, state_dict=...)`.

The architecture is deliberately simple and NOT a transformer: a two-layer GRU
over a learned token embedding, followed by a linear LM head. Two tokenizer
backends are supported (selected per model id):

  - `byte`: a tiny deterministic UTF-8 byte tokenizer (vocab 259). The head is
    ~0.2M params, total model ~11M params — small enough that MeZO's projected
    gradient actually moves the loss.
  - `qwen`: the Qwen3-0.6B tokenizer (vocab ~151k). The head is ~155M params and
    total model ~185M params.
"""
from dataclasses import dataclass
import json
import os

import torch
import torch.nn as nn
from transformers import BatchEncoding


@dataclass
class DecoderOutput:
    last_hidden_state: torch.Tensor


@dataclass
class LMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class ByteTokenizer:
    """Minimal UTF-8 byte-level tokenizer exposing the HF API subset ZOTitan uses.

    Tokens 0..255 are raw UTF-8 bytes; 256/257/258 are pad/eos/unk. It only needs
    to support `pad_token_id`, `tokenizer(text, ...)["input_ids"]` (list) and
    `tokenizer(text, return_tensors="pt", ...).to(device)` (2-D tensors).
    """
    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        self.pad_token_id = 256
        self.eos_token_id = 257
        self.unk_token_id = 258
        self.vocab_size = 259

    def __len__(self):
        return self.vocab_size

    def __call__(self, text, truncation=False, max_length=None, return_tensors=None, **kw):
        single = isinstance(text, str)
        texts = [text] if single else list(text)
        # Slice the bytes object before materializing the int list: TinyStories
        # stories average ~900 bytes against a max_length of 256, so building the
        # full list first would throw ~70% of it away.
        limit = max_length if truncation else None
        encs = [list(t.encode("utf-8")[:limit]) for t in texts]

        if return_tensors == "pt":
            max_len = max((len(e) for e in encs), default=0)
            id_t = torch.tensor([e + [self.pad_token_id] * (max_len - len(e)) for e in encs],
                                dtype=torch.long)
            attn = torch.tensor([[1] * len(e) + [0] * (max_len - len(e)) for e in encs],
                                dtype=torch.long)
            return BatchEncoding({"input_ids": id_t, "attention_mask": attn})

        if single:
            return BatchEncoding({"input_ids": encs[0], "attention_mask": [1] * len(encs[0])})
        return BatchEncoding({"input_ids": encs,
                              "attention_mask": [[1] * len(e) for e in encs]})

    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "byte_tokenizer.json"), "w") as f:
            json.dump({"vocab_size": self.vocab_size, "pad_token_id": self.pad_token_id}, f)


class TinyStoryDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, hidden: int,
                 num_layers: int, pad_token_id: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.rnn = nn.GRU(d_model, hidden, num_layers=num_layers, batch_first=True)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        # attention_mask is accepted for HF API compatibility but ignored: the pad
        # token's embedding is zeroed via padding_idx, so padded positions contribute
        # nothing (only a little recurrent bias drift), and the CE loss masks them.
        x = self.embed(input_ids)
        out, _ = self.rnn(x)
        return DecoderOutput(last_hidden_state=out)


class TinyStoryLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, hidden: int = 1024,
                 num_layers: int = 2, pad_token_id: int = 0):
        super().__init__()
        self.decoder = TinyStoryDecoder(vocab_size, d_model, hidden, num_layers, pad_token_id)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)

    # ── HF API surface used by losses.py ─────────────────────────────────────
    def get_decoder(self):
        return self.decoder

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                labels: torch.Tensor | None = None) -> LMOutput:
        hidden = self.decoder(input_ids, attention_mask).last_hidden_state
        logits = self.lm_head(hidden)          # (B, T, V)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
            shift_labels = labels[:, 1:].reshape(-1)
            loss = nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
        return LMOutput(logits=logits, loss=loss)

    # ── HF API surface used by the checkpoint path ───────────────────────────
    def save_pretrained(self, save_directory: str, state_dict: dict | None = None, **kwargs):
        """Write a plain torch state dict (ZOTitan's checkpoint path, not HF reloadable)."""
        os.makedirs(save_directory, exist_ok=True)
        if state_dict is None:
            state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_directory, "pytorch_model.bin"))


# model_id -> (tokenizer kind, architecture kwargs). vocab_size/pad_token_id come
# from the tokenizer. Add more rows here for new toy architectures — this table is
# the single source of truth for which local model ids exist (model.load_model
# dispatches off it).
TINY_ARCHS = {
    "zotitan/tinystories-rnn":      dict(tokenizer="qwen", d_model=128, hidden=1024, num_layers=2),
    "zotitan/tinystories-byte-rnn": dict(tokenizer="byte",  d_model=256, hidden=1024, num_layers=2),
}


def _build_tokenizer(kind: str):
    """Return (tokenizer, vocab_size, pad_token_id) for a tokenizer kind."""
    if kind == "byte":
        tok = ByteTokenizer()
        return tok, tok.vocab_size, tok.pad_token_id
    if kind == "qwen":
        from transformers import AutoTokenizer
        from .model import DEFAULT_MODEL_ID
        tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL_ID)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        return tok, len(tok), tok.pad_token_id
    raise ValueError(f"unknown tokenizer kind {kind!r}")


def build_tiny(model_id: str, hidden: int | None = None, num_layers: int | None = None) -> tuple[TinyStoryLM, object]:
    """Build (model, tokenizer) for a tiny-model id. Device/dtype are the caller's job."""
    arch = TINY_ARCHS.get(model_id)
    if arch is None:
        raise ValueError(f"unknown tiny model {model_id!r}; available: {sorted(TINY_ARCHS)}")
    kw = {k: v for k, v in arch.items() if k != "tokenizer"}
    if hidden is not None:
        kw["hidden"] = hidden
    if num_layers is not None:
        kw["num_layers"] = num_layers
    tokenizer, vocab_size, pad_token_id = _build_tokenizer(arch["tokenizer"])
    return TinyStoryLM(vocab_size=vocab_size, pad_token_id=pad_token_id, **kw), tokenizer

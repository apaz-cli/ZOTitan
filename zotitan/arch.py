"""Local architectures: a registry of pure-torch LMs for pretraining experiments.

An architecture is selected by an arch spec in `--model.model-id`, using the same
component grammar as objectives (see spec.py / the README):

    --model.model-id gru
    --model.model-id "(transformer,d_model=512,n_layers=12)"
    --model.model-id "(zotitan/tinystories-byte-rnn,num_layers=3)"

Any model id that doesn't name a registered arch falls through to HuggingFace.

Adding one: subclass `ZOLM`, implement `body()` (and build `self.embed` /
`self.lm_head`), decorate with `@register_arch("name")`, and drop the file in
`archs/`. Nothing is required to be differentiable: ZO only ever runs the forward.
If the forward draws random numbers (sampled routing, noisy writes, ...), use the
default torch generator — the ZO loop reseeds it so the +εz and -εz passes of a pair
see the same draws (see ZOOptimizer.run_forward_passes).

`ZOLM` provides the slice of the HF `PreTrainedModel` surface ZOTitan consumes:
  - the reference loss path (losses._torch_xentropy) calls `model(**batch)` and reads
    `.loss` / `.logits`;
  - the fused Liger path (losses._liger_xentropy) calls `get_decoder()` and
    `get_output_embeddings()` and reads `.last_hidden_state` / `head.weight`;
  - the checkpoint path calls `model.save_pretrained(dir, state_dict=...)`.
"""
from dataclasses import dataclass
import json
import os
from typing import Any

import torch
import torch.nn as nn
from transformers import BatchEncoding

from .spec import parse_component


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


def build_tokenizer(kind: str):
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
    if kind == "noise":
        # The fixed BPE-4096 tokenizer of github.com/qlabs-eng/noise (MIT), vendored verbatim.
        from transformers import PreTrainedTokenizerFast
        path = os.path.join(os.path.dirname(__file__), "assets", "noise_bpe4096.json")
        tok = PreTrainedTokenizerFast(tokenizer_file=path, eos_token="<|endoftext|>",
                                      pad_token="<|endoftext|>")
        return tok, len(tok), tok.pad_token_id
    raise ValueError(f"unknown tokenizer kind {kind!r} (expected 'byte', 'qwen' or 'noise')")


class _DecoderView:
    """What `get_decoder()` hands the fused loss path: calling it runs the body."""
    def __init__(self, lm: "ZOLM"):
        self._lm = lm

    def __call__(self, input_ids, attention_mask=None):
        return DecoderOutput(last_hidden_state=self._lm.body(input_ids, attention_mask))


class ZOLM(nn.Module):
    """Base class for local LMs. Subclasses set `self.embed` (nn.Embedding) and
    `self.lm_head` (nn.Linear, no bias) and implement `body`. The constructor must
    accept `vocab_size` and `pad_token_id` keyword arguments (filled in from the
    tokenizer) plus whatever architecture kwargs the spec passes."""
    embed: nn.Embedding
    lm_head: nn.Linear

    logit_softcap: float | None = None
    """If set, logits are capped as c·tanh(logits/c), on both loss paths."""

    def body(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        """(B, T) token ids -> (B, T, H) final hidden states, H = lm_head.in_features.
        Inputs are right-padded; positions past the mask are ignored by the loss."""
        raise NotImplementedError

    # ── HF API surface used by losses.py ─────────────────────────────────────
    def get_decoder(self):
        return _DecoderView(self)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                labels: torch.Tensor | None = None) -> LMOutput:
        logits = self.lm_head(self.body(input_ids, attention_mask))   # (B, T, V)
        if self.logit_softcap is not None:
            c = self.logit_softcap
            logits = c * torch.tanh(logits.float() / c)
        loss = None
        if labels is not None:
            # fp32: a bf16 CE is quantized to ~0.03 near ln(V), which rounds L+ and L- together.
            shift_logits = logits[:, :-1, :].float().reshape(-1, logits.size(-1))
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


# ── registry ──────────────────────────────────────────────────────────────────

# name -> (base name or class, preset kwargs). A class entry is a concrete arch; a
# str entry is a preset that resolves through another registered name.
ARCHS: dict[str, tuple[type[ZOLM] | str, dict[str, Any]]] = {}


def register_arch(name: str, **defaults):
    """Class decorator: register a ZOLM subclass under `name`. `defaults` are applied
    before the spec's kwargs (the spec wins). `tokenizer` (default 'byte') is consumed
    by the builder ('byte', 'qwen' or 'noise'), every other kwarg goes to the constructor."""
    def deco(cls):
        if name in ARCHS:
            raise ValueError(f"arch {name!r} is already registered")
        ARCHS[name] = (cls, defaults)
        return cls
    return deco


def register_preset(name: str, base: str, **kwargs):
    """Register `name` as `base` with `kwargs` pre-applied (the spec still overrides)."""
    if name in ARCHS:
        raise ValueError(f"arch {name!r} is already registered")
    ARCHS[name] = (base, kwargs)


def _resolve(name: str) -> tuple[type[ZOLM], dict[str, Any]]:
    """Follow preset links down to a class, merging kwargs (outermost preset wins)."""
    kwargs: dict[str, Any] = {}
    seen = []
    while True:
        if name not in ARCHS:
            raise ValueError(f"unknown arch {name!r}; available: {sorted(ARCHS)}")
        if name in seen:
            raise ValueError(f"preset cycle: {' -> '.join(seen + [name])}")
        seen.append(name)
        target, kw = ARCHS[name]
        kwargs = {**kw, **kwargs}
        if not isinstance(target, str):
            return target, kwargs
        name = target


def is_arch_spec(model_id: str) -> bool:
    """True if `model_id` names a registered arch (bare or as a `(name,k=v)` spec)."""
    try:
        name, _, _ = parse_component(model_id)
    except ValueError:
        return False
    return name in ARCHS


def build_arch(spec: str) -> tuple[ZOLM, object]:
    """Build (model, tokenizer) from an arch spec. Device/dtype are the caller's job."""
    name, positional, spec_kwargs = parse_component(spec)
    if positional:
        raise ValueError(f"arch specs take only KEY=VALUE fields, got {positional} in {spec!r}")
    cls, kwargs = _resolve(name)
    kwargs = {**kwargs, **spec_kwargs}
    tokenizer, vocab_size, pad_token_id = build_tokenizer(kwargs.pop("tokenizer", "byte"))
    return cls(vocab_size=vocab_size, pad_token_id=pad_token_id, **kwargs), tokenizer


def print_param_report(model: nn.Module):
    """Print total params split into input embedding, output head, and body (the rest).
    Compare architectures at matched *body* size: the head and embedding are fixed by
    the vocab, and the ZO estimator's SNR falls with the trainable count."""
    total = sum(p.numel() for p in model.parameters())
    seen: set[int] = set()

    def count(mod) -> int:
        if mod is None:
            return 0
        n = 0
        for p in mod.parameters():
            if id(p) not in seen:              # tied embeddings are counted once
                seen.add(id(p))
                n += p.numel()
        return n

    embed = count(getattr(model, "get_input_embeddings", lambda: None)())
    head  = count(getattr(model, "get_output_embeddings", lambda: None)())
    print(f"  params: {total:,} total | body {total - embed - head:,} | "
          f"embed {embed:,} | head {head:,}")


# Import the concrete archs for their @register_arch side effects. Kept at the bottom
# because those modules import names defined above.
from . import archs  # noqa: E402,F401

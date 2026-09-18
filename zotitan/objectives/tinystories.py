"""The TinyStories causal-LM objective (registry name "tinystories").

A small, self-contained pretraining objective over `roneneldan/TinyStories`.
The dataset is already cached by HF in this environment, but the objective
streams/lazy-loads it rather than downloading a snapshot up front, so
`dataset_sources()` is empty.
"""
import math
import random

import torch

from ..data import EVAL_SAMPLES, batched
from ..objective import (CrossEntropyCriterion, RubricObjective, register_objective)


@register_objective("tinystories")
class TinyStoriesObjective(RubricObjective):
    DATASET_ID = "roneneldan/TinyStories"

    CHUNK = 512
    """Rows are read and tokenized this many stories at a time: one `ds[indices]` take
    plus one batched tokenizer call is several times cheaper than a Python loop of
    single-row lookups, and the shuffled order is identical either way."""

    def __init__(self, compile_enabled: bool, compile_mode: str | None,
                 max_seq_len: int, num_train: int | None = None, **kwargs):
        """`num_train` caps how many stories are used per epoch (None = all ~2.1M).
        Remaining kwargs configure the CE loss (fused, z_loss_weight, ...)."""
        self.max_seq_len = max_seq_len
        self.num_train   = num_train
        self._ds         = None
        super().__init__([(CrossEntropyCriterion(compile_enabled, compile_mode, **kwargs), 1.0)])

    # ── data ──
    def dataset_sources(self):
        # TinyStories is loaded lazily from the HF cache; nothing to download up front.
        return []

    def _load(self):
        if self._ds is None:
            from datasets import load_dataset
            self._ds = load_dataset(self.DATASET_ID, split="train")
        return self._ds

    # ── train / eval ──
    def train_batches(self, tokenizer, seed: int, batch_size: int):
        """Infinite iterator of shuffled, right-padded causal-LM batches."""
        ds = self._load()
        n  = len(ds) if self.num_train is None else min(self.num_train, len(ds))
        rng = random.Random(seed)

        def pairs():
            idx = list(range(n))
            while True:
                rng.shuffle(idx)
                for s in range(0, n, self.CHUNK):
                    texts = ds[idx[s:s + self.CHUNK]]["text"]
                    for ids in tokenizer(texts, truncation=True,
                                         max_length=self.max_seq_len)["input_ids"]:
                        # TinyStories has a handful of empty/blank rows; skipping
                        # anything shorter than one input→target pair covers them
                        # too, so the RNN never sees a zero-length sequence.
                        if len(ids) >= 2:
                            yield ids, ids

        return batched(pairs(), tokenizer.pad_token_id, batch_size)

    def evaluate(self, model, tokenizer, n_examples=None, split=None) -> dict[str, float]:
        from datasets import load_dataset
        n = EVAL_SAMPLES if n_examples is None else n_examples
        val_ds = load_dataset(self.DATASET_ID, split="validation")
        model.eval()
        device = next(model.parameters()).device
        total_loss, total_tokens = 0.0, 0
        print(f"  computing ppl over {n} examples...")
        with torch.no_grad():
            for i, item in enumerate(val_ds):
                if i >= n:
                    break
                enc = tokenizer(item["text"], return_tensors="pt",
                                max_length=self.max_seq_len, truncation=True).to(device)
                if enc.input_ids.shape[1] < 2:
                    continue
                loss = model(**enc, labels=enc.input_ids).loss
                k = enc.input_ids.shape[1] - 1
                total_loss += loss.item() * k
                total_tokens += k
        return {"ppl": math.exp(total_loss / total_tokens)}

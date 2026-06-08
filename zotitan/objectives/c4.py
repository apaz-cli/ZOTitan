"""The C4 (causal-LM perplexity) objective (registry name "c4")."""
import math
import os
import torch

from ..data import EVAL_SAMPLES, batched
from ..objective import (CrossEntropyCriterion, DatasetSource, RubricObjective, register_objective)


@register_objective("c4")
class C4Objective(RubricObjective):
    # The ~50 GB local subset (allenai/c4 'en', load_from_disk format) is identified by
    # NAME, not a path: it lives under the HF datasets cache (see _dir), so there's no
    # hardcoded location — only HF_HOME / HF_DATASETS_CACHE (or a data_dir override) decide.
    SUBSET_NAME  = "zotitan-c4-en-50g"
    SUBSET_BYTES = 50 * 1024**3       # stream train until ~this many text bytes, then stop

    def __init__(self, compile_enabled: bool, compile_mode: str | None,
                 max_seq_len: int, data_dir: str | None = None, **kwargs):
        # `data_dir` overrides where the subset lives; None ⇒ HF datasets cache. The
        # remaining kwargs configure the CE loss.
        self.max_seq_len = max_seq_len
        self._data_dir = data_dir
        super().__init__([(CrossEntropyCriterion(compile_enabled, compile_mode, **kwargs), 1.0)])

    # ── dataset prep ──
    def _dir(self) -> str:
        if self._data_dir is not None:
            return self._data_dir
        from datasets import config
        return os.path.join(config.HF_DATASETS_CACHE, self.SUBSET_NAME)

    def _build(self) -> None:
        """Stream allenai/c4 'en' into _dir(): train capped at SUBSET_BYTES, validation full.
        BOTH splits stream so only the shards we read download — a non-streaming
        load_dataset(split=...) would resolve the whole ~300 GB config first."""
        from datasets import load_dataset, Dataset
        d = self._dir()
        def stream_split(split: str, byte_budget: int | None) -> None:
            def gen():
                total = 0
                for ex in load_dataset("allenai/c4", "en", split=split, streaming=True):
                    yield ex
                    if byte_budget is not None:
                        total += len(ex["text"].encode("utf-8"))
                        if total >= byte_budget:
                            break
            print(f"[c4] streaming {split} → {d}/{split} ...")
            Dataset.from_generator(gen).save_to_disk(f"{d}/{split}")
        stream_split("train", self.SUBSET_BYTES)
        stream_split("validation", None)
        print(f"[c4] done → {d}")

    def dataset_sources(self) -> list[DatasetSource]:
        d = self._dir()
        is_present = lambda: os.path.isdir(f"{d}/train") and os.path.isdir(f"{d}/validation")
        return [DatasetSource(self.SUBSET_NAME, is_present, self._build)]

    # ── train / eval ──
    def train_batches(self, tokenizer, seed: int, batch_size: int):
        """One pass over the shuffled local C4 train split; causal-LM labels = input_ids
        (padding masked to -100 by pad_batch)."""
        from datasets import load_from_disk, Dataset
        ds = load_from_disk(f"{self._dir()}/train").shuffle(seed=seed)
        assert isinstance(ds, Dataset)
        def pairs():
            for item in ds:
                assert isinstance(item, dict) and "text" in item
                ids = tokenizer(item["text"], truncation=True, max_length=self.max_seq_len)["input_ids"]
                yield ids, ids
        return batched(pairs(), tokenizer.pad_token_id, batch_size)

    def evaluate(self, model, tokenizer, n_examples=None, split=None) -> dict[str, float]:
        from datasets import load_from_disk, Dataset
        n = EVAL_SAMPLES if n_examples is None else n_examples
        val_ds = load_from_disk(f"{self._dir()}/validation")
        assert isinstance(val_ds, Dataset)
        val_ds.select(range(n))
        model.eval()
        total_loss, total_tokens = 0.0, 0
        device = next(model.parameters()).device
        print(f"  computing ppl over {n} examples...")
        with torch.no_grad():
            for i, item in enumerate(val_ds):
                if i % 5000 == 0:
                    print(f"  ppl: {i}/{n}")
                assert isinstance(item, dict) and "text" in item
                enc = tokenizer(item["text"], return_tensors="pt",
                                max_length=self.max_seq_len, truncation=True).to(device)
                if enc.input_ids.shape[1] < 2:
                    continue
                loss = model(**enc, labels=enc.input_ids).loss
                k = enc.input_ids.shape[1] - 1
                total_loss += loss.item() * k
                total_tokens += k
        return {"ppl": math.exp(total_loss / total_tokens)}

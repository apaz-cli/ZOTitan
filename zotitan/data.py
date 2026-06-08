import threading
import torch
from typing import Iterator

# Shared training-data primitives. Dataset-specific loading/eval lives on the objectives
# (SciJudgeObjective, C4Objective in objectives/); what remains here is generic: the
# batch collator and the async GPU prefetch loop.

EVAL_SAMPLES = 1_000


def pad_batch(examples: list[tuple[list, list]], pad_id: int) -> dict:
    """Right-pad a list of (input_ids, labels) into a rectangular tensor-dict batch
    (labels padded with -100, attention_mask 1 on real tokens / 0 on padding)."""
    max_len = max(len(inp) for inp, _ in examples)
    input_ids, labels, attention_mask = [], [], []
    for inp, lab in examples:
        pad = max_len - len(inp)
        input_ids.append(inp + [pad_id] * pad)
        labels.append(lab + [-100] * pad)
        attention_mask.append([1] * len(inp) + [0] * pad)
    return {
        "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
        "labels":         torch.tensor(labels,          dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask,  dtype=torch.long),
    }


def batched(pairs: Iterator[tuple[list, list]], pad_id: int, batch_size: int) -> Iterator[dict]:
    """Group an iterable of (input_ids, labels) examples into padded tensor-dict batches."""
    batch: list[tuple[list, list]] = []
    for pair in pairs:
        batch.append(pair)
        if len(batch) == batch_size:
            yield pad_batch(batch, pad_id)
            batch = []


# ── async prefetch ────────────────────────────────────────────────────────────

class PrefetchLoader:
    """Wrap an iterator of dict-of-tensor batches; yield each already on `device`.

    A background thread fetches the next batch (the loader's tokenization/padding
    runs there) and copies it to the GPU on a side stream, while the main thread
    consumes the previous batch. One batch of lookahead, single thread.

    Loosely adapted from PrimeIntellect's PrefetchDataLoader
    (github.com/PrimeIntellect-ai/prime-diloco), stripped to the essentials: no
    torch DataLoader, no stateful checkpointing.
    """

    def __init__(self, iterator, device, move_fn=None):
        self._it     = iter(iterator)
        self.device  = torch.device(device)
        self._stream = torch.cuda.Stream(self.device) if self.device.type == "cuda" else None
        self._move   = move_fn          # custom mover (e.g. mixtures); None = built-in flat-dict path
        self._ready  = None
        self._thread = None
        self._prefetch()

    def _move_batch(self, batch):
        if self._move is not None:
            return self._move(batch, self.device)
        # Built-in fast path for plain tensor dicts: pin so the copy is truly async.
        return {k: v.pin_memory().to(self.device, non_blocking=True)
                for k, v in batch.items()}

    def _prefetch(self):
        def _task():
            try:
                batch = next(self._it)
            except BaseException as e:  # StopIteration or a real loader error
                self._ready = e
                return
            if self._stream is not None:
                # run the copy on a side stream so it overlaps the default stream.
                with torch.cuda.stream(self._stream):
                    batch = self._move_batch(batch)
            elif self._move is not None:
                batch = self._move(batch, self.device)
            else:
                batch = {k: v.to(self.device) for k, v in batch.items()}
            self._ready = batch

        self._thread = threading.Thread(target=_task, daemon=True)
        self._thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        assert self._thread is not None, "PrefetchLoader not properly initialized"
        self._thread.join()
        item, self._ready = self._ready, None
        if isinstance(item, BaseException):
            raise item
        if self._stream is not None:
            # make the consuming (default) stream wait for the side-stream copies.
            torch.cuda.current_stream(self.device).wait_stream(self._stream)
        self._prefetch()
        return item


def get_batches(loader, n: int) -> list[dict] | None:
    """Pull n batches from loader. Returns the full list, or None if the loader
    runs dry before n are collected (signals end of training)."""
    batches = []
    for _ in range(n):
        batch = next(loader, None)
        if batch is None:
            return None
        batches.append(batch)
    return batches

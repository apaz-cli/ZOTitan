"""The Noise speedrun objective (registry name "noise"): fixed data, fixed score.

Ported from github.com/qlabs-eng/noise (MIT, (c) 2026 qlabs-eng), `utils/prepare_data.py`
and `utils/eval.py`:

  data   the Dust paper's FineWeb split, pre-tokenized with the fixed BPE-4096 tokenizer:
         4,880 train rows and 544 val rows of 2,049 tokens (2,048 predictions each).
         Rebuilt on first use and checked against their pinned hashes.
  train  ONE epoch over the train rows (their rule: each row feeds at most one update),
         in their order at seed 0. The iterator ends after the epoch, so training stops.
  score  mean next-token CE over all 1,114,112 val predictions (`val_loss`). Their
         records: EGGROLL 5.954, Dust+Adam 4.167; backprop SGD 5.007, Adam 3.769.

Use with `--model.model-id noise_gpt` (or any arch built with `tokenizer='noise'`). Their
methods score the whole population on one shared batch of 8 rows per update (610 updates);
the equivalent here is `--zo.shared-batch --zo.batch-size 8`. Without shared_batch, MeZO
consumes z_batch batches per step and the epoch ends after 4880 / (batch_size·z_batch) steps.
"""
import hashlib
import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ..objective import DatasetSource, RubricObjective, register_objective
from ..schedule import maybe_torchcompile

SEQ        = 2049
TRAIN_ROWS = 4880        # 9,994,240 prediction tokens
VAL_ROWS   = 544         # 1,114,112 prediction tokens
SKIP_DOCS  = 200_000     # the first FineWeb docs trained the BPE-4096 tokenizer
POOL_TOKENS     = 20_500_000
VAL_POOL_TOKENS = 1_150_000
VOCAB      = 4096

# sha256 of the int32 little-endian token bytes, pinned upstream.
EXPECTED = {
    "train.pt": "36ccb57eba6f5474b18acf99edd4a4793187260f7da7cc7053322cce59154bde",
    "val.pt":   "5fdb9242ce8ab83e519ab5dd5cf8d64f4d8a99f6e74c137f4ca0f723da5f7437",
}


def token_hash(rows: torch.Tensor) -> str:
    return hashlib.sha256(rows.to(torch.int32).contiguous().numpy().tobytes()).hexdigest()


def load_rows(path: str, rows: int) -> torch.Tensor:
    """Load a fixed split as int64 [rows, 2049]; refuses anything but the pinned data."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, torch.Tensor) or tuple(data.shape) != (rows, SEQ):
        raise ValueError(f"{path}: expected a [{rows}, {SEQ}] token tensor")
    if token_hash(data) != EXPECTED[os.path.basename(path)]:
        raise ValueError(f"{path}: token hash mismatch; delete it to rebuild")
    return data.long()


class NextTokenCriterion:
    """Mean CE on pre-shifted rows: input_ids = row[:-1], targets = row[1:], so all 2,048
    predictions of a row count without running attention over 2,049 positions."""
    name = "ce"
    differentiable = True

    def __init__(self, compile_enabled: bool = False, compile_mode: str | None = None):
        self.fn = maybe_torchcompile(_next_token_ce, enabled=compile_enabled, mode=compile_mode)

    def __call__(self, model, batch):
        # Clone before the next call: under cudagraphs the ZO loop holds the + pass's loss
        # (lazily, no sync) while the - pass replays the graph over the same output buffer.
        loss = self.fn(model, batch["input_ids"], batch["targets"]).clone()
        return loss, {"ce": loss}


def _next_token_ce(model, input_ids, targets):
    logits = model(input_ids=input_ids).logits
    return F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())


@register_objective("noise")
class NoiseObjective(RubricObjective):
    SUBSET_NAME = "zotitan-noise"

    def __init__(self, compile_enabled: bool, compile_mode: str | None,
                 max_seq_len: int, data_dir: str | None = None, eval_batch_size: int = 16):
        """`data_dir` overrides where train.pt/val.pt live (None = HF datasets cache).
        max_seq_len is ignored: the rows are fixed at 2,049 tokens."""
        self._data_dir = data_dir
        self.eval_batch_size = eval_batch_size
        super().__init__([(NextTokenCriterion(compile_enabled, compile_mode), 1.0)])

    # ── dataset prep ──
    def _dir(self) -> str:
        if self._data_dir is not None:
            return self._data_dir
        from datasets import config
        return os.path.join(config.HF_DATASETS_CACHE, self.SUBSET_NAME)

    def _build(self) -> None:
        """Their prepare_data.py: FineWeb sample-10BT in stream order, skip SKIP_DOCS, a
        ~20.5M-token pool whose first 4,880 rows are train, then the val documents. Docs
        are tokenized in batches but consumed one at a time, so the split points (and the
        hashes) match the upstream one-doc-at-a-time loop exactly."""
        from datasets import load_dataset
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(os.path.join(os.path.dirname(__file__), "..", "assets",
                                                     "noise_bpe4096.json"))
        end = tokenizer.token_to_id("<|endoftext|>")
        corpus = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
        print(f"[noise] skipping the first {SKIP_DOCS:,} FineWeb documents ...")
        # Skip by iterating, as upstream does: IterableDataset.skip() lands elsewhere in the
        # stream and the hashes then fail to match.
        texts = (ex["text"] for i, ex in enumerate(corpus) if i >= SKIP_DOCS)

        def encoded():
            while True:
                chunk = [t for _, t in zip(range(1024), texts)]
                if not chunk:
                    return
                yield from (e.ids for e in tokenizer.encode_batch(chunk))

        docs = encoded()

        def encode_split(min_tokens: int) -> torch.Tensor:
            tokens: list[int] = []
            while len(tokens) < min_tokens:
                tokens.append(end)
                tokens.extend(next(docs))
            usable = len(tokens) // SEQ * SEQ
            return torch.tensor(tokens[:usable], dtype=torch.int32).view(-1, SEQ)

        print("[noise] tokenizing the train pool ...")
        train = encode_split(POOL_TOKENS)[:TRAIN_ROWS].clone()
        print("[noise] tokenizing the val pool ...")
        val = encode_split(VAL_POOL_TOKENS)[:VAL_ROWS].clone()

        d = self._dir()
        os.makedirs(d, exist_ok=True)
        for name, rows in (("train.pt", train), ("val.pt", val)):
            digest = token_hash(rows)
            if digest != EXPECTED[name]:
                raise RuntimeError(f"[noise] {name}: sha256 {digest} does not match the pinned "
                                   f"{EXPECTED[name]}; not saving")
            torch.save(rows, os.path.join(d, name))
        with open(os.path.join(d, "manifest.json"), "w") as f:
            json.dump({"source": "github.com/qlabs-eng/noise utils/prepare_data.py",
                       "files": {n: EXPECTED[n] for n in EXPECTED}}, f, indent=2)
        print(f"[noise] done → {d}")

    def dataset_sources(self) -> list[DatasetSource]:
        d = self._dir()
        is_present = lambda: all(os.path.isfile(os.path.join(d, n)) for n in EXPECTED)
        return [DatasetSource(self.SUBSET_NAME, is_present, self._build)]

    def _check_tokenizer(self, tokenizer):
        if len(tokenizer) != VOCAB:
            raise ValueError(f"the noise data is BPE-4096 tokens; the model's tokenizer has "
                             f"{len(tokenizer)} — build the arch with tokenizer='noise'")

    # ── train / eval ──
    def train_batches(self, tokenizer, seed: int, batch_size: int):
        """One epoch over the train rows in a seeded permutation (seed 0 = their order)."""
        self._check_tokenizer(tokenizer)
        if dist.is_initialized() and dist.get_world_size() > 1:
            # dp ranks get distinct data seeds, so they would each walk their own epoch and
            # reuse rows across updates, breaking the one-epoch rule.
            raise NotImplementedError("the noise objective's one-epoch loader is single-rank only")
        rows = load_rows(os.path.join(self._dir(), "train.pt"), TRAIN_ROWS)
        rows = rows[torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed + 1))]
        for i in range(0, len(rows) - batch_size + 1, batch_size):
            b = rows[i:i + batch_size]
            yield {"input_ids": b[:, :-1].contiguous(), "targets": b[:, 1:].contiguous()}

    @torch.no_grad()
    def evaluate(self, model, tokenizer, n_examples=None, split=None) -> dict[str, float]:
        """Their score: mean loss over all 1,114,112 val predictions (n_examples caps rows)."""
        self._check_tokenizer(tokenizer)
        model.eval()
        device = next(model.parameters()).device
        rows = load_rows(os.path.join(self._dir(), "val.pt"), VAL_ROWS)
        if n_examples is not None:
            rows = rows[:n_examples]
        total = torch.zeros(2, dtype=torch.float64, device=device)   # summed loss, predictions
        for batch in rows.split(self.eval_batch_size):
            batch = batch.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=batch[:, :-1].contiguous()).logits
            total[0] += F.cross_entropy(logits.float().flatten(0, 1), batch[:, 1:].flatten(),
                                        reduction="sum").double()
            total[1] += batch[:, 1:].numel()
        value = (total[0] / total[1]).item()
        if value != value:
            raise FloatingPointError("non-finite validation loss")
        return {"val_loss": value}

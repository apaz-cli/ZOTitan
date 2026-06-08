"""The Countdown objective (registry name "countdown").

A generative, reward-scored task (ZO-only, `differentiable = False`): given numbers and a
target, the model writes a chain-of-thought and an `<answer>` equation, scored by a verifier
rather than a differentiable loss.

    reward = 0.1·format + answer        # answer dominates; format just shapes the output

`score` returns the CE-style loss `MAX_REWARD − reward` (0 ⇔ perfect), so it reads like the
other objectives. The constant offset is gradient-invariant under ZO, so only the logged
number changes, not the optimization.

Data is the HF dataset Jiayi-Pan/Countdown-Tasks-3to4 (just nums/target); the prompt is
rebuilt locally in `_context`. Train = front of the split, eval = the last `n_eval` rows.
"""
import random
import re
import torch

from ..data import EVAL_SAMPLES
from ..objective import DatasetSource, Score, register_objective

# ── verifier reward ───────────────────────────────────────────────────────────────
# Ported from es-fine-tuning-paper/countdown/countdown_task.py. `response` is the model's
# continuation after the prompt's trailing "<think>" (we score only the generated tokens).

_ANSWER_RE      = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_FULL_FORMAT_RE = re.compile(r"^<think>.*?</think>\n<answer>.*?</answer>$", re.DOTALL)
_THINK_RE       = re.compile(r"<think>.*?</think>", re.DOTALL)
_ALLOWED_CHARS  = re.compile(r"^[0-9+\-*/() ]+$")


def _format_reward(response: str) -> float:
    """1.0 for the exact <think>…</think>\\n<answer>…</answer> shape, else partial credit."""
    response = "<think>" + response
    if _FULL_FORMAT_RE.match(response):
        return 1.0
    return 0.1 * bool(_THINK_RE.search(response)) + 0.5 * bool(_ANSWER_RE.search(response))


def _answer_reward(response: str, nums: list[int], target: int) -> float:
    """1.0 iff the last <answer> uses each number once and evaluates to target."""
    matches = _ANSWER_RE.findall(response)
    if not matches or not _ALLOWED_CHARS.match(matches[-1]):
        return 0.0
    answer = matches[-1]
    if sorted(int(n) for n in re.findall(r"\d+", answer)) != sorted(nums):
        return 0.0
    try:
        result = eval(answer, {"__builtins__": None}, {})   # arithmetic only (chars gated above)
        return 1.0 if abs(float(result) - target) < 1e-5 else 0.0
    except Exception:
        return 0.0


def _reward(response: str, nums: list[int], target: int) -> tuple[float, float, float]:
    """(total, answer, format) with total = 0.1·format + answer."""
    fmt = _format_reward(response)
    ans = _answer_reward(response, nums, target)
    return 0.1 * fmt + ans, ans, fmt


MAX_REWARD = 0.1 + 1.0   # perfect format + answer; the loss floor


# ── prompt ────────────────────────────────────────────────────────────────────────
# The TinyZero scaffold; ends mid-turn at "<think>" so the model continues straight into
# its reasoning. nums render as "[44 19 35]" (space-separated).
_PROMPT = (
    "You are a helpful assistant. You first think about the reasoning process in your "
    "mind and then provide the user with the answer."
    "Using the numbers {nums}, create an equation that equals {target}. You can use "
    "basic arithmetic operations (+, -, *, /) and each number can only be used once. "
    "Show your work in <think> </think> tags. And return the final answer in <answer> "
    "</answer> tags, for example <answer> (1 + 2) / 3 </answer>."
    "Let me solve this step by step.\n<think>"
)


def _context(nums: list[int], target: int) -> str:
    return _PROMPT.format(nums="[" + " ".join(map(str, nums)) + "]", target=target)


# ── objective ─────────────────────────────────────────────────────────────────────

@register_objective("countdown")
class CountdownObjective:
    DATASET_ID = "Jiayi-Pan/Countdown-Tasks-3to4"
    CACHE_DIR  = None

    differentiable = False               # reward over generated text — ZO-only
    flat_batches   = False               # batches carry non-tensor metadata (nums/target)

    def __init__(self, compile_enabled: bool, compile_mode: str | None,
                 max_seq_len: int, max_new_tokens: int = 1024, n_eval: int = EVAL_SAMPLES,
                 deterministic: int = 0):
        # compile_* are required by the factory signature but unused (no CE kernel here).
        # deterministic: if non-zero, pass this int as the seed to
        #   transformers.enable_full_determinism (0 → disabled).
        self.max_seq_len    = max_seq_len
        self.max_new_tokens = max_new_tokens
        self.n_eval         = n_eval
        if deterministic:
            from transformers import enable_full_determinism
            enable_full_determinism(deterministic)
        self._ds            = None        # cached dataset
        self._tokenizer     = None        # captured in train_batches for score()

    # ── data ──
    def _load(self):
        if self._ds is None:
            from datasets import load_dataset
            self._ds = load_dataset(self.DATASET_ID, split="train", cache_dir=self.CACHE_DIR)
        return self._ds

    def dataset_sources(self) -> list[DatasetSource]:
        from huggingface_hub import snapshot_download
        kw = {"repo_id": self.DATASET_ID, "repo_type": "dataset", "cache_dir": self.CACHE_DIR}
        def is_present() -> bool:
            try:
                snapshot_download(**kw, local_files_only=True)
                return True
            except Exception:
                return False
        return [DatasetSource(self.DATASET_ID, is_present, lambda: snapshot_download(**kw))]

    def _split_indices(self) -> tuple[list[int], list[int]]:
        """(train, eval) row indices; eval is the last n_eval rows."""
        n = len(self._load())
        cut = max(0, n - self.n_eval)
        return list(range(cut)), list(range(cut, n))

    # ── batching ──
    def _collate(self, tokenizer, rows: list[dict]) -> dict:
        """Left-pad the prompts for generation; keep nums/target alongside."""
        enc = tokenizer([_context(r["nums"], r["target"]) for r in rows], return_tensors="pt",
                        padding=True, padding_side="left", truncation=True, max_length=self.max_seq_len)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "nums": [list(r["nums"]) for r in rows], "target": [int(r["target"]) for r in rows]}

    def to_device(self, batch, device):
        return {**batch, "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device)}

    def _rewards(self, model, tokenizer, batch) -> tuple[list[float], list[float], list[float]]:
        """Greedy-generate, decode the new tokens, and score each. Returns the reward lists."""
        ids, attn = batch["input_ids"], batch["attention_mask"]
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        out = model.generate(input_ids=ids, attention_mask=attn, do_sample=False,
                             use_cache=True,
                             max_new_tokens=self.max_new_tokens, pad_token_id=pad_id)
        texts = tokenizer.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)
        rows = [_reward(t, n, tg) for t, n, tg in zip(texts, batch["nums"], batch["target"])]
        totals, answers, formats = zip(*rows)
        return list(totals), list(answers), list(formats)

    # ── train / score / eval ──
    def train_batches(self, tokenizer, seed: int, batch_size: int):
        """Infinite stream of shuffled train batches."""
        self._tokenizer = tokenizer
        ds = self._load()
        train_idx, _ = self._split_indices()
        def gen():
            rng = random.Random(seed)
            while True:
                rng.shuffle(train_idx)
                for s in range(0, len(train_idx) - batch_size + 1, batch_size):
                    yield self._collate(tokenizer, [ds[int(i)] for i in train_idx[s:s + batch_size]])
        return gen()

    def score(self, model, batch) -> Score:
        totals, answers, formats = self._rewards(model, self._tokenizer, batch)
        n = len(totals)
        mean = sum(totals) / n
        value = torch.tensor(MAX_REWARD - mean, device=batch["input_ids"].device)  # 0 ⇔ perfect
        return Score(value, {"reward": mean, "answer": sum(answers) / n, "format": sum(formats) / n})

    def evaluate(self, model, tokenizer, n_examples=None, split=None) -> dict[str, float]:
        """Mean reward / answer-accuracy / format over the held-out eval tail."""
        ds = self._load()
        _, eval_idx = self._split_indices()
        eval_idx = eval_idx[:EVAL_SAMPLES if n_examples is None else n_examples]
        model.eval()
        device = next(model.parameters()).device
        tot_r = tot_a = tot_f = 0.0
        print(f"  computing countdown reward over {len(eval_idx)} examples...")
        with torch.no_grad():
            for s in range(0, len(eval_idx), 16):
                rows = [ds[int(i)] for i in eval_idx[s:s + 16]]
                totals, answers, formats = self._rewards(
                    model, tokenizer, self.to_device(self._collate(tokenizer, rows), device))
                tot_r += sum(totals); tot_a += sum(answers); tot_f += sum(formats)
                done = min(s + 16, len(eval_idx))
                print(f"  reward: {done}/{len(eval_idx)}  acc={tot_a/done:.3f}")
        n = len(eval_idx)
        return {"reward": tot_r / n, "answer_acc": tot_a / n, "format": tot_f / n}

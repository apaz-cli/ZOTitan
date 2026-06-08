"""The SciJudgeBench objective (registry name "scijudge")."""
import json
import random
import torch

from ..data import EVAL_SAMPLES, batched
from ..objective import (CrossEntropyCriterion, DatasetSource, RubricObjective, register_objective)


@register_objective("scijudge")
class SciJudgeObjective(RubricObjective):
    # HF dataset (CACHE_DIR=None ⇒ default ~/.cache/huggingface). test_ood_iclr is
    # excluded: schema mismatch in HF datasets + a different task (acceptance vs citations).
    DATASET_ID = "OpenMOSS-Team/SciJudgeBench"
    CACHE_DIR  = None
    FILES = {"train": "train.jsonl", "test": "test.jsonl", "test_ood_year": "test_ood_year.jsonl"}

    def __init__(self, compile_enabled: bool, compile_mode: str | None,
                 max_seq_len: int, split: str | None = None, **kwargs):
        # `split` is this objective's own knob (the default eval split); the remaining
        # kwargs configure the CE loss.
        self.max_seq_len = max_seq_len
        self._split = split
        super().__init__([(CrossEntropyCriterion(compile_enabled, compile_mode, **kwargs), 1.0)])

    # ── dataset prep ──
    def _dir(self) -> str:
        """Local snapshot path, downloading (train + test splits) if needed."""
        from huggingface_hub import snapshot_download
        return snapshot_download(repo_id=self.DATASET_ID, repo_type="dataset", cache_dir=self.CACHE_DIR)

    def dataset_sources(self) -> list[DatasetSource]:
        from huggingface_hub import snapshot_download
        def is_present() -> bool:
            try:
                snapshot_download(repo_id=self.DATASET_ID, repo_type="dataset",
                                  cache_dir=self.CACHE_DIR, local_files_only=True)
                return True
            except Exception:
                return False
        return [DatasetSource(self.DATASET_ID, is_present, self._dir)]

    # ── tokenization ──
    # '<answer>{X}</answer>' tokenizes so '>' merges with the answer letter into one token
    # ('>A'/'>B'); every other token is identical for A and B. That merged token is the ONLY
    # position distinguishing the answers — the only token we teacher-force and the only logit
    # we read at eval. _prefix_and_decisions derives both together so train and eval can't drift.
    @staticmethod
    def _chat_ids(tokenizer, messages, add_generation_prompt: bool,
                  max_len: int | None = None) -> list[int]:
        # max_len set ⇒ truncate to it (leaving room for the appended decision scaffold).
        kw = {"truncation": True, "max_length": max_len - 8} if max_len is not None else {}
        out = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=add_generation_prompt, **kw)
        return out if isinstance(out, list) else list(out["input_ids"])  # list, or BatchEncoding when truncating

    @staticmethod
    def _prefix_and_decisions(tokenizer) -> tuple[list[int], dict[str, int]]:
        """Returns (prefix_ids, {letter: decision_id}): prefix is '<answer' WITHOUT the
        trailing '>' (so the letter merges forward into the decision token)."""
        prefix = tokenizer.encode("<answer>", add_special_tokens=False)[:-1]  # drop trailing '>'
        decisions = {}
        for letter in ("A", "B"):
            full = tokenizer.encode(f"<answer>{letter}</answer>", add_special_tokens=False)
            assert full[:len(prefix)] == prefix, "answer prefix tokenization drifted"
            decisions[letter] = full[len(prefix)]
        return prefix, decisions

    @classmethod
    def _example(cls, tokenizer, item, prefix: list[int], decisions: dict[str, int],
                 max_len: int) -> tuple[list, list]:
        """Supervise only the decision token; the shared scaffold is neither appended nor scored."""
        context_ids = cls._chat_ids(tokenizer, item["messages"], add_generation_prompt=True, max_len=max_len)
        decision    = decisions[item["correct_answer"]]
        input_ids = context_ids + prefix + [decision]
        labels    = [-100] * (len(context_ids) + len(prefix)) + [decision]
        return input_ids, labels

    # ── train / eval ──
    def train_batches(self, tokenizer, seed: int, batch_size: int):
        """Infinite iterator over the shuffled SciJudgeBench train split."""
        with open(f"{self._dir()}/{self.FILES['train']}") as f:
            lines = f.readlines()
        prefix, decisions = self._prefix_and_decisions(tokenizer)
        rng = random.Random(seed)
        while True:
            rng.shuffle(lines)
            pairs = (self._example(tokenizer, json.loads(l), prefix, decisions, self.max_seq_len) for l in lines)
            yield from batched(pairs, tokenizer.pad_token_id, batch_size)

    def evaluate(self, model, tokenizer, n_examples=None, split=None) -> dict[str, float]:
        """Answer-token accuracy: probe the exact transition training supervises — condition
        on the '<answer' prefix and compare the '>A' vs '>B' logits."""
        split = split or self._split or "test"
        n = EVAL_SAMPLES if n_examples is None else n_examples
        with open(f"{self._dir()}/{self.FILES[split]}") as f:
            lines = f.readlines()[:n]
        prefix, decisions = self._prefix_and_decisions(tokenizer)
        a_id, b_id = decisions["A"], decisions["B"]
        model.eval()
        device = next(model.parameters()).device
        desc, correct = f"acc/{split}", 0
        print(f"  computing {desc} over {len(lines)} examples...")
        with torch.no_grad():
            for i, line in enumerate(lines):
                item = json.loads(line)
                ctx  = self._chat_ids(tokenizer, item["messages"], add_generation_prompt=True)
                inp  = torch.tensor([ctx + prefix], dtype=torch.long, device=device)
                logits = model(input_ids=inp).logits[0, -1]
                if ("A" if logits[a_id] > logits[b_id] else "B") == item["correct_answer"]:
                    correct += 1
                if (i + 1) % 200 == 0:
                    print(f"  {desc}: {i+1}/{len(lines)}  acc={correct/(i+1):.3f}")
        return {"acc": correct / len(lines)}

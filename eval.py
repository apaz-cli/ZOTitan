import os
import tyro
import torch
from dataclasses import dataclass, field
from mlsweep.logger import MLSweepLogger

from model import load_model, ModelConfig
from objective import make_objective, prepare_datasets
from pretty import print_eval_results


@dataclass
class EvalConfig:
    objective: str = "scijudge"
    """Which objective/dataset to evaluate: scijudge | c4."""

    model: ModelConfig = field(default_factory=ModelConfig)
    """Model ID and init mode."""

    checkpoint: str = ""
    """Path to load after init. Accepts a HuggingFace model dir (has config.json)
    or a raw .pt state_dict file. Empty = pretrained weights as-is."""

    split: str = "test"
    """SciJudgeBench split to evaluate: test | test_ood_year"""

    n_examples: int = 1_000
    """Number of eval examples."""


def main():
    cfg = tyro.cli(EvalConfig)
    objective = make_objective(cfg.objective, max_seq_len=cfg.model.max_seq_len)
    prepare_datasets(objective)
    with MLSweepLogger() as logger:
        model, tokenizer = load_model(cfg.model)

        if cfg.checkpoint:
            if os.path.isdir(cfg.checkpoint):
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(
                    cfg.checkpoint, torch_dtype=torch.bfloat16, device_map="cuda"
                )
            else:
                state = torch.load(cfg.checkpoint, map_location="cuda", weights_only=True)
                model.load_state_dict(state)
            print(f"  loaded checkpoint: {cfg.checkpoint}")

        results = objective.evaluate(model, tokenizer, n_examples=cfg.n_examples, split=cfg.split)
        logger.log({**results, "split": cfg.split})
        print_eval_results(results, split=cfg.split)


if __name__ == "__main__":
    main()

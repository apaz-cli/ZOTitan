"""Smoke tests: 5-step FO and ZO runs for full finetune + every LoRA strategy."""
import pytest
from mlsweep.logger import MLSweepLogger

from zotitan.train import ExperimentConfig, TrainingConfig, run
from zotitan.train_fo import FOConfig
from zotitan.train_zo import ZOConfig
from zotitan.lora import LoRAConfig
from zotitan.schedule import BaseTrainConfig

# Parameterize so it will be fast to launch by disabling torchcompile

@pytest.mark.parametrize("optimizer,strategy", [
    ("fo", "none"), ("fo", "standard"), ("fo", "relora"), ("fo", "continual"),
    ("zo", "none"), ("zo", "standard"), ("zo", "relora"), ("zo", "continual"),
])
def test_smoke(optimizer, strategy):
    cfg = ExperimentConfig(
        optimizer=optimizer,
        lora=LoRAConfig(strategy=strategy, relora_cycles=2),
        training=TrainingConfig(steps=5, eval=False,
                                compile_mode=("default" if optimizer == "zo" and strategy == "standard" else "none"),
                                compile_cudagraphs=False),
        fo=FOConfig(batch_size=2, base=BaseTrainConfig(ckpt_every=999_999)),
        zo=ZOConfig(batch_size=2, base=BaseTrainConfig(lr=1e-3, ckpt_every=999_999)),
    )
    with MLSweepLogger() as logger:
        run(cfg, logger)

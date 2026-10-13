"""The noise objective: pinned data, the one-epoch loader, and the score."""
import os

import pytest
import torch

from zotitan.model import ModelConfig, load_model
from zotitan.objective import make_objective
from zotitan.objectives.noise import TRAIN_ROWS

obj = make_objective("noise")
pytestmark = pytest.mark.skipif(not all(s.is_present() for s in obj.dataset_sources()),
                                reason="noise data not built (run prepare_datasets)")


def test_one_epoch_in_upstream_order():
    _, tok = load_model(ModelConfig(model_id="noise_gpt"))
    batches = list(obj.train_batches(tok, seed=0, batch_size=8))
    assert len(batches) == TRAIN_ROWS // 8 == 610
    rows = torch.cat([torch.cat([b["input_ids"], b["targets"][:, -1:]], 1) for b in batches])
    assert len(torch.unique(rows, dim=0)) == TRAIN_ROWS            # every row once
    # Upstream's order: randperm(seed + 1) over train.pt.
    from zotitan.objectives.noise import load_rows
    ref = load_rows(os.path.join(obj._dir(), "train.pt"), TRAIN_ROWS)
    ref = ref[torch.randperm(TRAIN_ROWS, generator=torch.Generator().manual_seed(1))]
    assert torch.equal(rows, ref)


def test_score_at_init_is_uniform():
    model, tok = load_model(ModelConfig(model_id="noise_gpt"))
    val = obj.evaluate(model, tok, n_examples=16)["val_loss"]
    assert abs(val - torch.log(torch.tensor(4096.0)).item()) < 0.1


def test_wrong_tokenizer_rejected():
    _, tok = load_model(ModelConfig(model_id="gru"))
    with pytest.raises(ValueError):
        next(obj.train_batches(tok, seed=0, batch_size=8))

"""Local arch registry, shared ± forward randomness, and ZO on a non-differentiable arch."""
import pytest
import torch

from zotitan.arch import ARCHS, build_arch, is_arch_spec
from zotitan.lora import LoRAConfig
from zotitan.model import ModelConfig, load_model
from zotitan.objective import CrossEntropyCriterion
from zotitan.parallel import ParallelConfig, init_mesh
from zotitan.schedule import BaseTrainConfig
from zotitan.train import ExperimentConfig, TrainingConfig, run
from zotitan.train_zo import MomentumConfig, ZOConfig, ZOOptimizer


def _body(model) -> int:
    return sum(p.numel() for n, p in model.named_parameters()
               if not n.startswith(("embed.", "lm_head.")))


def test_spec_resolution():
    assert is_arch_spec("gru") and is_arch_spec("(transformer,n_layers=2)")
    assert not is_arch_spec("Qwen/Qwen3-0.6B-Base")
    # The preset keeps the old tiny-model body size the findings were measured at.
    model, tok = build_arch("zotitan/tinystories-byte-rnn")
    assert tok.vocab_size == 259 and _body(model) == 10_235_904
    # Spec kwargs override the preset.
    deeper, _ = build_arch("(zotitan/tinystories-byte-rnn,num_layers=3)")
    assert deeper.rnn.num_layers == 3
    with pytest.raises(ValueError):
        build_arch("(gru,4)")                    # arch specs are KEY=VALUE only
    with pytest.raises(ValueError):
        build_arch("no_such_arch")


@pytest.mark.parametrize("name", sorted(n for n, (t, _) in ARCHS.items() if not isinstance(t, str)))
def test_forward_shapes(name):
    """Every registered arch serves both loss paths (reference and fused Liger)."""
    model, tok = load_model(ModelConfig(model_id=name))
    ids = torch.randint(0, tok.vocab_size, (2, 16), device="cuda")
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids}
    with torch.inference_mode():
        out = model(**batch)
        assert out.logits.shape == (2, 16, tok.vocab_size) and torch.isfinite(out.loss)
        for fused in (False, True):
            val, _ = CrossEntropyCriterion(fused=fused)(model, batch)
            assert torch.isfinite(val)


def test_pm_passes_share_forward_noise():
    """With z = 0 the ± passes evaluate the same weights, so they agree exactly iff the
    stochastic forward drew the same noise; different pairs still draw different noise."""
    mesh = init_mesh(ParallelConfig())
    model, tok = load_model(ModelConfig(model_id="hardroute"))
    model = model.float()                        # bf16 CE at init is too coarse to see the noise
    named = [(n, p) for n, p in model.named_parameters()]
    opt = ZOOptimizer(named, ZOConfig(batch_size=2), None, mesh)
    ids = torch.randint(0, tok.vocab_size, (2, 32), device="cuda")
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids}
    score = CrossEntropyCriterion(fused=False)
    score_fn = lambda m, b: type("S", (), {"value": score(m, b)[0], "metrics": {}})()
    zero = [torch.zeros_like(p) for _, p in named]

    (pos, _), (neg, _) = opt.run_forward_passes(model, batch, zero, score_fn, fwd_seed=1)
    (other, _), _      = opt.run_forward_passes(model, batch, zero, score_fn, fwd_seed=2)
    assert pos.item() == neg.item()
    assert pos.item() != other.item()            # the noise is real, not a no-op


def test_zo_trains_nondifferentiable_arch():
    """ZO overfits one batch through hard argmax routing (no gradient reaches the router)."""
    losses = []

    class Logger:
        def log(self, m):
            if "loss" in m:
                losses.append(m["loss"])
        def sync(self):
            pass

    cfg = ExperimentConfig(
        optimizer="zo",
        objective="(tinystories,fused=False)",
        model=ModelConfig(model_id="hardroute", max_seq_len=64),
        lora=LoRAConfig(strategy="none"),
        training=TrainingConfig(steps=60, eval=False, compile_mode="none", compile_cudagraphs=False),
        zo=ZOConfig(batch_size=8, z_batch=8, eps=1e-3,
                    mom=MomentumConfig(momentum_method="stored_ema", second_moment=True),
                    base=BaseTrainConfig(lr=3e-3, ckpt_every=999_999, overfit_first_batch=True)),
    )
    run(cfg, Logger())
    assert len(losses) == 60
    assert sum(losses[-10:]) / 10 < sum(losses[:10]) / 10 - 0.3, losses

"""Data parallelism on 2 GPUs against a single-GPU reference.

Each test runs the real trainers twice on a tiny random fp32 Qwen3 and synthetic data,
once on 2 ranks and once on 1 rank fed the equivalent global batches. Workers only save
their results; the asserts run here.

  pair-sharded ZO (dp | z_batch): ranks draw distinct z; replicas agree up to the residue
                                  of their own perturbation round trips; matches z_batch
                                  pairs on one GPU.
  batch-sharded ZO (z_batch = 1): ranks draw the same z on distinct data fragments;
                                  matches the concatenated batch on one GPU.
  FO (dp = 2):                    replicas are identical; the averaged gradient of the
                                  fragments is the full batch's gradient.

Ranks may sit on different GPU models, so cross-run comparisons use a tolerance.
"""
import os
import socket

import pytest
import torch
import torch.multiprocessing as mp
from mlsweep.logger import MLSweepLogger

from zotitan.objective import CrossEntropyCriterion, RubricObjective
from zotitan.parallel import DATA_SEED_STRIDE, ParallelConfig, average_grads, init_mesh
from zotitan.schedule import BaseTrainConfig
from zotitan.train_fo import FOConfig, train_fo
from zotitan.train_zo import ZOConfig, ZOOptimizer, train_zo

pytestmark = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")

VOCAB, SEQ, STEPS, SEED = 128, 32, 3, 0


def tiny_qwen() -> torch.nn.Module:
    """Tiny random Qwen3, built on CPU so every rank inits alike."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=SEQ, tie_word_embeddings=True)
    cfg._attn_implementation = "eager"
    return Qwen3ForCausalLM(cfg).float()


class SyntheticLM(RubricObjective):
    """Random-token causal LM. `ranks`/`local_z` make it serve, on one GPU, the global
    batches that 2 ranks would load. With local_z set they are pair-sharded (each step's
    pairs are rank 0's local_z batches, then rank 1's). With local_z=0 they are
    batch-sharded (fragments concatenated)."""
    name = "synthetic"

    def __init__(self, ranks: int = 1, local_z: int = 0):
        super().__init__([(CrossEntropyCriterion(fused=True, compute_accuracy=False), 1.0)])
        self.ranks, self.local_z = ranks, local_z

    def train_batches(self, tokenizer, seed, batch_size):
        bs      = batch_size if self.local_z else batch_size // self.ranks
        streams = [self._stream(seed + DATA_SEED_STRIDE * r, bs) for r in range(self.ranks)]
        while True:
            if self.local_z:
                for s in streams:
                    for _ in range(self.local_z):
                        yield next(s)
            else:
                parts = [next(s) for s in streams]
                yield {k: torch.cat([p[k] for p in parts]) for k in parts[0]}

    @staticmethod
    def _stream(seed, batch_size):
        g = torch.Generator().manual_seed(seed)
        while True:
            ids = torch.randint(0, VOCAB, (batch_size, SEQ), generator=g)
            yield {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}


def zo_cfg(**kw):
    return ZOConfig(eps=1e-3, base=BaseTrainConfig(lr=1e-3, ckpt_every=10**9, loss_kill_threshold=0), **kw)


def params(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()}


def first_z(cfg, mesh, model):
    """The z this rank perturbs with for its first pair of step 1."""
    named = list(model.named_parameters())
    opt   = ZOOptimizer(named, cfg, lambda b: None, mesh)
    seeds = opt._draw_seeds(cfg.z_batch)
    z     = opt._sample_z(seeds[opt._pair_lo])
    return {n: t.cpu() for (n, _), t in zip(named, z)}


# ── workers train, then save this rank's results to out/rank<r>.pt ────────────

def _zo_worker(out, cfg, objective, dp):
    mesh  = init_mesh(ParallelConfig(dp=dp))
    model = tiny_qwen().cuda()
    res   = {"z": first_z(cfg, mesh, model)}
    with MLSweepLogger() as logger:
        train_zo(model, None, STEPS, SEED, None, logger, cfg, objective, mesh=mesh)
    res["params"] = params(model)
    torch.save(res, os.path.join(out, f"rank{torch.distributed.get_rank()}.pt"))


def _fo_worker(out, cfg, objective, dp):
    mesh  = init_mesh(ParallelConfig(dp=dp))
    model = tiny_qwen().cuda()
    with MLSweepLogger() as logger:
        train_fo(model, None, STEPS, SEED, None, logger, cfg, objective, mesh)
    res = {"params": params(model)}

    # One more backward from the initial weights, with train_fo's gradient averaging.
    model = tiny_qwen().cuda()
    dp    = mesh["dp"]
    batch = next(objective.train_batches(None, SEED + DATA_SEED_STRIDE * dp.get_local_rank(),
                                         cfg.batch_size // dp.size()))
    objective.score(model, objective.to_device(batch, "cuda")).value.backward()
    average_grads(list(model.parameters()), mesh)
    res["grads"] = {n: p.grad.cpu() for n, p in model.named_parameters() if p.grad is not None}
    torch.save(res, os.path.join(out, f"rank{torch.distributed.get_rank()}.pt"))


def _launch(rank, world, port, fn, args):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world),
                      MASTER_ADDR="localhost", MASTER_PORT=str(port))
    fn(*args)


def spawn(world, fn, out, *args) -> list[dict]:
    os.makedirs(out)
    with socket.socket() as s:
        s.bind(("localhost", 0))
        port = s.getsockname()[1]
    mp.spawn(_launch, args=(world, port, fn, (out, *args)), nprocs=world, join=True)
    return [torch.load(os.path.join(out, f"rank{r}.pt")) for r in range(world)]


def assert_close(got, ref, what, atol=1e-5):
    for n, t in got.items():
        err = (t - ref[n]).abs().max().item()
        assert err < atol, f"{what}: {n} is off by {err:.3e}"


def assert_identical(a, b, what):
    for n, t in a.items():
        assert torch.equal(t, b[n]), f"{what}: {n} differs"


# ── tests ─────────────────────────────────────────────────────────────────────

def test_zo_dp_pair_sharded(tmp_path):
    cfg       = zo_cfg(z_batch=4, batch_size=2)
    r0, r1    = spawn(2, _zo_worker, str(tmp_path / "dp"), cfg, SyntheticLM(), 2)
    ref,      = spawn(1, _zo_worker, str(tmp_path / "ref"), cfg, SyntheticLM(ranks=2, local_z=2), 1)
    assert not any(torch.equal(r0["z"][n], r1["z"][n]) for n in r0["z"]), "ranks must draw distinct z"
    assert_close(r0["params"], r1["params"], "replicas")
    assert_close(r0["params"], ref["params"], "dp=2 vs one GPU")


def test_zo_dp_batch_sharded(tmp_path):
    cfg       = zo_cfg(z_batch=1, batch_size=4)
    r0, r1    = spawn(2, _zo_worker, str(tmp_path / "dp"), cfg, SyntheticLM(), 2)
    ref,      = spawn(1, _zo_worker, str(tmp_path / "ref"), cfg, SyntheticLM(ranks=2), 1)
    assert_identical(r0["z"], r1["z"], "batch-sharded z")
    assert_identical(r0["params"], r1["params"], "replicas")
    assert_close(r0["params"], ref["params"], "dp=2 vs one GPU")


def test_fo_dp(tmp_path):
    cfg       = FOConfig(batch_size=4, base=BaseTrainConfig(lr=1e-3, ckpt_every=10**9, loss_kill_threshold=0))
    r0, r1    = spawn(2, _fo_worker, str(tmp_path / "dp"), cfg, SyntheticLM(), 2)
    ref,      = spawn(1, _fo_worker, str(tmp_path / "ref"), cfg, SyntheticLM(ranks=2), 1)
    assert_identical(r0["params"], r1["params"], "replicas")
    # Params after AdamW aren't compared to the reference. Its early steps ≈ lr·g/|g| blow
    # summation-order noise in near-zero grads up to a sizeable fraction of a step.
    for n, g in ref["grads"].items():
        rel = ((r0["grads"][n] - g).norm() / g.norm()).item()
        assert rel < 1e-4, f"averaged grad of {n} is off by {rel:.2e} (relative)"

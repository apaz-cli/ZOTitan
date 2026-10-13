"""Data parallelism over a ("dp",) DeviceMesh.

Launch with torchrun, or mlsweep with GPUS_PER_RUN = dp and SET_DIST_ENV = True. A plain
single-GPU run gets a 1-rank mesh, so there is one code path.
"""
import atexit
import os
import torch
import torch.distributed as dist
from dataclasses import dataclass
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

DATA_SEED_STRIDE = 1_000_003
"""dp rank r loads data from seed + r·stride, so each rank reads its own fragment and rank 0
keeps the single-GPU order."""


@dataclass
class ParallelConfig:
    """dp must equal WORLD_SIZE env var."""
    dp: int = 1


def init_mesh(cfg: ParallelConfig) -> DeviceMesh:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if cfg.dp != world:
        raise ValueError(f"parallel.dp={cfg.dp} must equal WORLD_SIZE={world} env var")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if not dist.is_initialized():
        store = dist.HashStore() if world == 1 else None     # no rendezvous for a lone process
        dist.init_process_group("nccl", store=store, rank=int(os.environ.get("RANK", "0")), world_size=world)
        atexit.register(dist.destroy_process_group)
    return init_device_mesh("cuda", (cfg.dp,), mesh_dim_names=("dp",))


@torch.no_grad()
def sync_replicas(tensors, mesh: DeviceMesh):
    """Copy dp rank 0's `tensors` to the other replicas, so random inits agree."""
    if mesh["dp"].size() == 1:
        return
    for t in tensors:
        dist.broadcast(t, group=mesh["dp"].get_group(), group_src=0)


@torch.no_grad()
def average_grads(params: list[torch.nn.Parameter], mesh: DeviceMesh):
    """Average the grads over dp, one flattened all-reduce per dtype."""
    if mesh["dp"].size() == 1:
        return
    grads = [p.grad for p in params if p.grad is not None]
    for dtype in {g.dtype for g in grads}:
        same = [g for g in grads if g.dtype == dtype]
        flat = _flatten_dense_tensors(same)
        dist.all_reduce(flat, op=dist.ReduceOp.AVG, group=mesh["dp"].get_group())
        torch._foreach_copy_(same, _unflatten_dense_tensors(flat, same))


def gather_objects(obj, mesh: DeviceMesh) -> list:
    """Every dp rank's `obj`, in rank order."""
    if mesh["dp"].size() == 1:
        return [obj]
    out: list = [None] * mesh["dp"].size()
    dist.all_gather_object(out, obj, group=mesh["dp"].get_group())
    return out


def any_rank(flag: bool, mesh: DeviceMesh) -> bool:
    """True if `flag` is set on any dp rank."""
    if mesh["dp"].size() == 1:
        return flag
    t = torch.tensor(flag, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=mesh["dp"].get_group())
    return bool(t)

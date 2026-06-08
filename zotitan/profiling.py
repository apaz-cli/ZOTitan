import contextlib
import os
import time
from dataclasses import dataclass

import torch

# Baked-in one-shot schedule: skip step 1, prime on step 2, record steps 3-5.
_WAIT, _WARMUP, _ACTIVE = 1, 1, 3


@dataclass
class ProfilingConfig:
    path: str = ""
    """Save a PyTorch profiler trace here, then exit the training loop after the
    capture. Empty disables profiling. A bare filename (no directory) is written
    to the mlsweep run dir so it is picked up as an artifact; a directory gets a
    default `profile.json`. `.json` is appended if the path has no extension."""

trace_saved: bool = False

@contextlib.contextmanager
def maybe_enable_profiling(cfg, *, run_dir="."):
    if not cfg.path:
        yield None
        return

    out = cfg.path
    if os.path.isdir(out) or out.endswith(os.sep):
        out = os.path.join(out, "profile.json")
    if not os.path.splitext(out)[1]:
        out += ".json"
    if not os.path.dirname(out):
        out = os.path.join(run_dir, out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    def trace_handler(prof):
        begin = time.monotonic()
        prof.export_chrome_trace(out)
        print(f"  profiler: saved trace to {out} ({time.monotonic() - begin:.2f}s)")
        global trace_saved
        trace_saved = True

    gpu = torch.profiler.ProfilerActivity.CUDA if torch.cuda.is_available() else None
    activities = [torch.profiler.ProfilerActivity.CPU] + ([gpu] if gpu else [])
    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=_WAIT, warmup=_WARMUP, active=_ACTIVE, repeat=1),
        on_trace_ready=trace_handler,
        record_shapes=True,
    ) as prof:
        yield prof

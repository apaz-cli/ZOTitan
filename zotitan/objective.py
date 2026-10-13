"""Objectives: a dataset + a scalar scorer + an eval, behind one interface.

An *objective* is the unit the train loops consume. It bundles three things that
are really one concept — "a scalar function of (model, data)":

  - train_batches: produce training batches
  - score:         (model, batch) -> Score(value, metrics); the thing to MINIMIZE
  - evaluate:      the same idea aggregated over a held-out split

ZO can optimize any scalar (differentiable or not, RL-style); FO needs the value
to be a differentiable tensor (`differentiable=True`). The loss *kernel* (fused vs
torch, z_loss) is an orthogonal axis a differentiable objective consumes via
`CrossEntropyCriterion`, not part of the objective's identity.

Objectives compose: `ObjectiveMixture([MixtureComponent(obj, loss_weight,
data_weight), ...])` is itself an Objective (composite pattern), so the train loops
never know whether they hold an atomic objective or a mixture.
"""
import fcntl
import json
import math
import os
import random
import time
import torch
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterator, Protocol

from .spec import parse_component, parse_value, split_top_level


# ── score / criterion contracts ────────────────────────────────────────────────

# value may be a 0-dim tensor. A bare value is averaged when steps or ranks are merged;
# tag it as (value, op) to reduce with another op from REDUCE instead.
Metric = torch.Tensor | float | tuple[torch.Tensor | float, str]

@dataclass
class Score:
    value:   torch.Tensor | float        # scalar to MINIMIZE (Tensor if differentiable)
    metrics: dict[str, Metric]           # everything to log (components of `value`, aux)


REDUCE: dict[str, Callable[[list[float]], float]] = {
    "mean": lambda xs: sum(xs) / len(xs), "sum": sum, "max": max, "min": min,
}


def floatify(metrics: dict[str, Metric]) -> dict[str, tuple[float, str]]:
    """Collapse a metrics dict to (float, op) pairs (one .item() per tensor)."""
    out: dict[str, tuple[float, str]] = {}
    for k, v in metrics.items():
        v, op = v if isinstance(v, tuple) else (v, "mean")
        if v is None:
            continue
        out[k] = (v.item() if isinstance(v, torch.Tensor) else float(v), op)
    return out


def merge_metrics(dicts: list[dict[str, tuple[float, str]]]) -> dict[str, float]:
    """Reduce each key over the floatified dicts that carry it (a mixture's steps, or
    ranks, may report different keys), with the key's op."""
    vals: dict[str, list[float]] = {}
    ops:  dict[str, str]         = {}
    for d in dicts:
        for k, (v, op) in d.items():
            vals.setdefault(k, []).append(v)
            ops[k] = op
    return {k: REDUCE[ops[k]](vs) for k, vs in vals.items()}


class Criterion(Protocol):
    """A single scalar term (≈ a verifiers reward fn): (model, batch) -> (value, metrics)."""
    name: str
    differentiable: bool
    def __call__(self, model, batch) -> tuple[torch.Tensor | float, dict[str, Metric]]: ...


class CrossEntropyCriterion:
    """Differentiable cross-entropy term over the model's LM head.

    value = ce + λ·z_loss (the scalar to minimize); metrics carries the components
    separately ({"ce", "z_loss", "acc"}). Holds only config and forwards to
    losses.cross_entropy, which picks the fused vs reference kernel (and imports the
    fused Liger kernel only when actually scored, so eval-only paths never pull it in).

    z_loss_weight:    coefficient λ for the z_loss regularizer λ · mean(logsumexp(logits)²).
                      Returned separately from the CE (not summed in). 0.0 disables it. The
                      "z" here is the softmax normalizer Z = Σ exp(logits) — unrelated to the
                      MeZO perturbation vector `z` in train_zo.py.
    fused:            use Liger Kernel's fused linear cross-entropy (faster, less memory).
    compute_accuracy: also return per-batch token-level argmax accuracy in `extra`.
    """
    name = "ce"
    differentiable = True

    def __init__(self, compile_enabled: bool = False, compile_mode: str | None = None,
                 z_loss_weight: float = 0.0, fused: bool = True, compute_accuracy: bool = True):
        from .losses import get_xentropy
        self.compile_enabled  = compile_enabled
        self.compile_mode     = compile_mode
        self.z_loss_weight    = z_loss_weight
        self.fused            = fused
        self.compute_accuracy = compute_accuracy
        self.fn               = get_xentropy(fused=self.fused, compile_enabled=self.compile_enabled,
                                              compile_mode=self.compile_mode)

    def __call__(self, model, batch):
        loss, z_loss, extra = self.fn(model, batch, self.z_loss_weight, self.compute_accuracy) # type: ignore
        if self.compile_enabled and not self.fused:
            # Clone before the next call: under cudagraphs the ZO loop holds the + pass's
            # outputs (lazily, no sync) while the - pass replays over the same buffers.
            loss   = loss.clone()
            z_loss = z_loss.clone() if z_loss is not None else None
            extra  = {k: v.clone() for k, v in extra.items()} if extra else extra
        value = loss if z_loss is None else loss + z_loss
        metrics: dict[str, Metric] = {"ce": loss}
        if z_loss is not None:
            metrics["z_loss"] = z_loss
        if extra:
            metrics.update(extra)            # e.g. {"acc": tensor}
        return value, metrics


# ── dataset preparation ───────────────────────────────────────────────────────────

@dataclass
class DatasetSource:
    """One downloadable dataset behind a named lock.

    key:        stable identity (e.g. the HF repo id) — used as BOTH the dedup key and
                the flock filename, so concurrent runs serialize per dataset.
    is_present: cheap local check; True ⇒ already downloaded, nothing to do.
    download:   perform the (possibly slow) download.

    Objectives advertise the sources they need via Objective.dataset_sources(); the
    startup driver `prepare_datasets` flocks per source so two runs on the same box can
    fetch DIFFERENT datasets at once instead of queueing behind one global lock."""
    key:        str
    is_present: Callable[[], bool]
    download:   Callable[[], object]    # return value (if any) is ignored — called for its side effect


def prepare_datasets(objective: "Objective", poll_interval: float = 2.0) -> None:
    """Download every dataset `objective` needs, parallel-safe across co-located runs.

    For each DatasetSource we try a NON-blocking exclusive flock. A source whose lock is
    held (another run is fetching it) is skipped this pass and retried on the next wrap-
    around, so concurrent runs grab whatever's free rather than serializing on one lock.
    Inside the lock we re-check presence (another run may have just finished), download if
    needed, then release. We only sleep when a whole pass found everything still locked."""
    pending = list({s.key: s for s in objective.dataset_sources()}.values())   # dedup by key
    while pending:
        still: list[DatasetSource] = []
        for s in pending:
            lock_path = f"/tmp/zotitan_prepare_{s.key.replace('/', '--')}.lock"
            fh = open(lock_path, "w")
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                fh.close()
                still.append(s)                       # another run holds it; retry later
                continue
            try:
                if s.is_present():
                    print(f"[prepare] {s.key}: already present.")
                else:
                    print(f"[prepare] {s.key}: downloading ...")
                    s.download()
                    print(f"[prepare] {s.key}: done.")
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
                fh.close()
        if still and len(still) == len(pending):
            time.sleep(poll_interval)                 # nothing free this pass — wait then retry
        pending = still


# ── objective protocol ──────────────────────────────────────────────────────────

class Objective(Protocol):
    name: str
    differentiable: bool                 # may FO use it? (ZO ignores)
    flat_batches: bool                   # True => batches are plain tensor dicts (fast prefetch path)
    def dataset_sources(self) -> list[DatasetSource]: ...
    def train_batches(self, tokenizer, seed: int, batch_size: int) -> Iterator: ...
    def to_device(self, batch, device) -> object: ...   # batch is opaque (dict, or (i, dict) for mixtures)
    def score(self, model, batch) -> Score: ...
    def evaluate(self, model, tokenizer, n_examples: int | None = None,
                 split: str | None = None) -> dict[str, float]: ...


# ── objective registry ──────────────────────────────────────────────────────────
# Register a new objective by decorating its class with @register_objective("name").
# make_objective looks it up by name — there is no central dispatch function to edit.

OBJECTIVES: dict[str, type] = {}


def register_objective(name: str):
    """Class decorator: register an Objective under `name` (its spec selector)."""
    def deco(cls):
        if name in OBJECTIVES:
            raise ValueError(f"objective {name!r} is already registered to {OBJECTIVES[name]!r}")
        cls.name = name
        OBJECTIVES[name] = cls
        return cls
    return deco


# ── atomic objectives ───────────────────────────────────────────────────────────

class RubricObjective:
    """Base: a dataset + a weighted list of criteria (a rubric).

    score() = Σ weightᵢ·valueᵢ over criteria, with merged metrics. A single-criterion
    objective is the common case; pass more (criterion, weight) pairs for a
    multi-criterion rubric (keys are then name-prefixed to avoid collisions). `name` is
    set on the class by @register_objective."""
    flat_batches = True

    def __init__(self, criteria: list[tuple[Criterion, float]]):
        self.criteria    = criteria
        self.differentiable = all(c.differentiable for c, _ in criteria)
        self._prefix        = len(criteria) > 1   # name-prefix metric keys only for multi-criterion rubrics

    def dataset_sources(self) -> list[DatasetSource]:
        return []                                  # nothing to download (e.g. local-disk C4)

    def to_device(self, batch, device):
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()}

    def score(self, model, batch) -> Score:
        total: torch.Tensor | float | None = None
        metrics: dict[str, Metric] = {}
        for crit, w in self.criteria:
            v, m = crit(model, batch)
            contrib = v if w == 1.0 else v * w
            total = contrib if total is None else total + contrib
            for k, mv in m.items():
                metrics[f"{crit.name}/{k}" if self._prefix else k] = mv
        assert total is not None, "objective has no criteria"
        return Score(total, metrics)

    # train_batches / evaluate are dataset-specific (subclasses in objectives/).


# Atomic objectives live one-per-module in objectives/ and self-register via
# @register_objective; objective.py imports that package at the bottom of this file
# so the registry is populated for any consumer that imports `objective`.


# ── mixture ─────────────────────────────────────────────────────────────────────

@dataclass
class MixtureComponent:
    objective:   Objective
    loss_weight: float = 1.0   # multiplier on this objective's value when active; NORMALIZED
    data_weight: float = 1.0   # relative sampling frequency;                      NORMALIZED


def _unique_names(names: list[str]) -> list[str]:
    """Disambiguate duplicate objective names (e.g. two SciJudge objectives) so metric
    prefixes and eval keys don't silently collide: ['a','a','b'] -> ['a#0','a#1','b']."""
    dup = {n for n, c in Counter(names).items() if c > 1}
    seen: Counter = Counter()
    out = []
    for n in names:
        out.append(f"{n}#{seen[n]}" if n in dup else n)
        seen[n] += 1
    return out


class ObjectiveMixture:
    """A weighted mixture of objectives — itself an Objective (composite).

    Mixing is at the *data* level (like verifiers' EnvGroup) but with explicit
    weights instead of raw dataset sizes: each step samples a component by
    `data_weight` proportion, draws one of its batches, and routes scoring to it.
    Batches stay homogeneous, so ZO's two-point invariant holds trivially and the
    per-batch device move just recurses. `loss_weight` (normalized) scales the active
    component's value — useful to up/down-weight a task's gradient independent of how
    often it's seen."""
    flat_batches = False

    def __init__(self, components: list[MixtureComponent]):
        if not components:
            raise ValueError("ObjectiveMixture requires at least one component")
        self.components     = components
        self.differentiable = all(c.objective.differentiable for c in components)
        self._names         = _unique_names([c.objective.name for c in components])
        self.name           = "mix(" + ",".join(self._names) + ")"
        dw = [c.data_weight for c in components]; sdw = sum(dw)
        lw = [c.loss_weight for c in components]; slw = sum(lw)
        self._probs        = [w / sdw for w in dw]
        self._loss_weights = [w / slw for w in lw]

    def dataset_sources(self) -> list[DatasetSource]:
        return [s for c in self.components for s in c.objective.dataset_sources()]

    def train_batches(self, tokenizer, seed: int, batch_size: int) -> Iterator:
        rng = random.Random(seed)
        mk = lambda i: self.components[i].objective.train_batches(tokenizer, seed + 1 + i, batch_size)
        iters = [mk(i) for i in range(len(self.components))]
        idxs  = range(len(self.components))
        while True:
            i = rng.choices(idxs, weights=self._probs, k=1)[0]
            batch = next(iters[i], None)
            if batch is None:                       # finite source (e.g. C4 DataLoader) exhausted
                iters[i] = mk(i)
                batch = next(iters[i], None)
                if batch is None:
                    raise RuntimeError(f"mixture component {i} produced no batches")
            yield (i, batch)

    def to_device(self, batch, device):
        i, b = batch
        return (i, self.components[i].objective.to_device(b, device))

    def score(self, model, batch) -> Score:
        i, b = batch
        name = self._names[i]
        s    = self.components[i].objective.score(model, b)
        w    = self._loss_weights[i]
        value = s.value if w == 1.0 else s.value * w
        metrics: dict[str, Metric] = {f"{name}/{k}": v for k, v in s.metrics.items()}
        metrics[f"{name}/loss"] = s.value         # the unweighted sub scalar
        return Score(value, metrics)

    def evaluate(self, model, tokenizer, n_examples=None, split=None) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, c in zip(self._names, self.components):
            sub = c.objective.evaluate(model, tokenizer, n_examples=n_examples, split=split)
            for k, v in sub.items():
                out[f"{name}/{k}"] = v
        return out


# ── factory ─────────────────────────────────────────────────────────────────────

def make_objective(dataset: str, compile_enabled: bool = False, compile_mode: str | None = None,
                   max_seq_len: int | None = None, **kwargs) -> Objective:
    """Build one atomic objective by registry name (see @register_objective).

    `max_seq_len` is the model's truncation length, passed on to the objective; None falls
    back to the ModelConfig default. Extra `kwargs` from the spec are forwarded to the
    objective's constructor (e.g. the CE objectives pass them through as loss params); an
    objective that does not declare a given key raises a TypeError naturally."""
    cls = OBJECTIVES.get(dataset)
    if cls is None:
        raise ValueError(f"Unknown objective {dataset!r}; registered: {sorted(OBJECTIVES)}")
    if max_seq_len is None:
        from .model import ModelConfig
        max_seq_len = ModelConfig().max_seq_len
    return cls(compile_enabled, compile_mode, max_seq_len, **kwargs)


# ── declarative config (string spec) ──────────────────────────────────────────────

@dataclass
class ObjectiveConfig:
    dataset:     str
    loss_weight: float = 1.0
    data_weight: float = 1.0
    kwargs:      dict = field(default_factory=dict)   # extra per-objective config from the spec


def parse_objective_spec(spec: str) -> list[ObjectiveConfig]:
    """Parse a mixture spec into a list of ObjectiveConfig.

    See the "Objective spec grammar" section of the README for the full grammar,
    quoting rules, and examples. In brief: a '+'-joined list of components, each a
    NAME or (NAME, positional weights, KEY=VALUE kwargs); bare values are numeric/
    bool/None literals and strings are single-quoted.
    """
    objs: list[ObjectiveConfig] = []
    for raw in split_top_level(spec, "+"):
        part = raw.strip()
        if not part:
            raise ValueError(f"empty component in objective spec: {spec!r}")
        name, positional, kwargs = parse_component(part)
        if len(positional) > 2:
            raise ValueError(
                f"too many positional fields in objective component {part!r} "
                f"(expected NAME[,loss_weight[,data_weight]])")
        weights = [float(parse_value(f)) for f in positional]
        lw = weights[0] if len(weights) >= 1 else 1.0
        dw = weights[1] if len(weights) == 2 else 1.0
        objs.append(ObjectiveConfig(dataset=name, loss_weight=lw,
                                    data_weight=dw, kwargs=kwargs))
    return objs


def build_objective(objs: list[ObjectiveConfig], compile_enabled: bool = False,
                    compile_mode: str | None = None, max_seq_len: int | None = None) -> Objective:
    """Build one Objective from parsed configs: 1 component -> atomic, N -> ObjectiveMixture."""
    if not objs:
        raise ValueError("build_objective requires at least one ObjectiveConfig")
    comps = [MixtureComponent(make_objective(o.dataset, compile_enabled, compile_mode,
                                             max_seq_len, **o.kwargs),
                              loss_weight=o.loss_weight, data_weight=o.data_weight)
             for o in objs]
    return comps[0].objective if len(comps) == 1 else ObjectiveMixture(comps)


# ── register the concrete objectives ──────────────────────────────────────────────
# Imported for its @register_objective side effects: the package imports every module
# under objectives/, each of which self-registers (populates OBJECTIVES). Kept at the
# very bottom — those modules import names defined above (register_objective,
# _RubricObjective, CrossEntropyCriterion, DatasetSource), so everything they need exists
# by the time this runs.
from . import objectives  # noqa: E402,F401

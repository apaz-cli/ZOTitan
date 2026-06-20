import collections
import math
import os
import time
import torch
import triton
import triton.language as tl
from dataclasses import dataclass, field
from typing import Literal
from .data import PrefetchLoader, get_batches
from .pretty import print_config, print_zo_step
from . import profiling
from .profiling import ProfilingConfig, maybe_enable_profiling
from .schedule import wsd_value, wsd_is_constant, BaseTrainConfig, maybe_torchcompile
from .lora import save_merged_checkpoint


@triton.jit
def _philox_randn_kernel(out_ptr, seed, n, BLOCK: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.random.randn(seed, offs), mask=offs < n)


def _philox_randn(seed: int, n: int, device) -> torch.Tensor:
    out = torch.empty(n, dtype=torch.float32, device=device)
    _philox_randn_kernel[(triton.cdiv(n, 1024),)](out, seed, n, BLOCK=1024)  # type: ignore
    return out

@dataclass
class MomentumConfig:
    """First- and second-moment configuration for the ZO optimizer.

    momentum_method selects the first-moment backend; second_moment adds an Adam-style
    second moment on top. The two compose orthogonally:

      momentum_method | second_moment | buffers | optimizer
      ----------------|---------------|---------|---------------------------------------
      none            | off           | 0       | MeZO (plain ZO-SGD)
      none            | on            | 1 (v)   | ZO-RMSProp
      stored_ema      | off           | 1 (m)   | MeZO-momentum
      stored_ema      | on            | 2 (m,v) | ZO-Adam
      seed_window     | off           | 0       | MeZO-momentum (memory-free, m reconstructed)
      seed_window     | on            | 1 (v)   | ZO-Adam (memory-reduced: m reconstructed, v stored)

    The second moment v is ALWAYS materialized, even when the seed_window (reconstructed m) first moment is not.
    Reconstructing v from seeds the way m is reconstructed is not viable: g² is nonlinear
    so it does not flatten into the per-seed weighted sum (it has within-step cross terms),
    and β₂'s ~1/(1-β₂) horizon (≈1000 steps at the default) cannot be faithfully captured
    by the short seed window. A reconstructed v would be a very compute-heavy approximation,
    enough not to be feasible.
    """

    momentum_method: Literal["none", "stored_ema", "seed_window"] = "none"
    """How the first-moment direction is maintained.
    none:        no momentum.
    stored_ema:  full EMA vector updated every step. Uses as much memory as the trainable params.
    seed_window: memory-efficient reconstruction from a circular buffer of RNG snapshots.
                 Does not use memory, but recomputes the z distributions from seeds over a step window."""

    beta1: float = 0.9
    """First-moment EMA decay (active when momentum_method="stored_ema"|"seed_window")"""

    beta2: float = 0.999
    """Second-moment EMA decay (active when second_moment=True). Adam's β₂."""

    second_moment: bool = False
    """Track a materialized EMA of grad_est² and divide the update by the bias-corrected
    sqrt(v) per dim (Adam/RMSProp denominator). Composes with any momentum_method (see table)."""

    denom_eps: float = 1e-8
    """Denominator stabilizer for the bias-corrected EMA and the sqrt(v) second-moment divide."""


@dataclass
class SeedWindowConfig:
    size: int = 0
    """Number of past seeds to retain in the circular buffer.
    0 disables seed-window momentum."""


@dataclass
class ClipConfig:
    """Clipping of the projected-gradient scalar (L₊-L₋)/2ε."""

    strategy: Literal["none", "per_pair", "norm"] = "none"
    """How the threshold τ is applied to a step's projected gradients:
    none:     no clipping.
    per_pair: clamp each pair's proj_grad to ±τ (per-sample; saturating → signSGD regime).
    norm:     scale the whole step's proj_grad vector by min(1, τ√Z/‖pg‖₂) so its RMS ≤ τ
              (the FO grad-norm-clip analog; preserves relative pair magnitudes).
    For z_batch=1 the two strategies coincide."""

    clip_pct: float = 0.0
    """Fraction of the highest |proj_grad| to clip, e.g. 0.05 clips the top 5% (τ = the
    95th percentile of recent |proj_grad|). 0 disables clipping regardless of strategy."""

    window: int = 1000
    """Number of recent |proj_grad| samples used to estimate the quantile threshold τ.
    The estimate lags by one step (built from history, applied to the current step) so a
    single step's own outliers cannot raise the threshold that clips them."""


@dataclass
class PerturbationConfig:
    distribution: Literal["gaussian", "low_rank", "polar"] = "gaussian"
    """Perturbation distribution.
    gaussian:  standard MeZO.
    low_rank:  LOZO.
    polar:     ZO-Muon (polar-orthogonal)."""

    rank: int = 64
    """Projection rank r for the low-rank factorization z = P @ Psi.
    ZO-Muon also uses a low-rank factorization for its perturbations, but with
    polar orthogonalization on the factors, so it is used both for low_rank and polar."""


@dataclass
class WeightDecayConfig:
    strategy: Literal["none", "standard", "anchored"] = "none"
    """Weight decay strategy.
    none:     no weight decay.
    standard: decay each parameter toward zero. θ -= lr * λ * θ.
    anchored: Anchored Weight Decay (AWD) — decay toward the initial model parameters.
              θ -= lr * λ * l'(θ - θ₀) where l' is the derivative of the penalty function."""

    lambda_: float = 0.0
    """Weight decay strength λ. Multiplied by the learning rate at each step.
    For standard: the coefficient of decay toward zero.
    For anchored: the AWD penalty factor."""

    penalty: Literal["l2", "l1"] = "l2"
    """Penalty function l(·) for anchored weight decay only.
    l2: ℓ₂ penalty — l'(x)=x, smooth shrinkage toward anchor θ₀.
    l1: ℓ₁ penalty — l'(x)=sign(x), sparse deviation from anchor θ₀."""


@dataclass
class FitnessShapingConfig:
    strategy: Literal["none", "centered_rank", "grpo"] = "none"
    """Fitness shaping over the per-step population of perturbations (OpenAI-ES or GRPO).

    none: plain ZO slope (MeZO).
    centered_rank:  OpenAI-ES. Sort the perturbation population and map to [-0.5, 0.5] by position in the list.
    grpo:           Group-relative normalization (like GRPO)

    Both transforms are only meaningful when every member is scored on the same data, so they require
    ZOConfig.shared_batch=True. Also mutually exclusive with and can be seen as an alternative to
    grad clipping.
    """

    normalize_std: bool = True
    """Only valid with grpo. Divide each advantage by the population's std so the update's size
    doesn't depend on reward scale."""


@dataclass
class ZOConfig:
    base: BaseTrainConfig = field(
        default_factory=lambda: BaseTrainConfig(lr=1e-5, ckpt_every=5_000)
    )

    # ── zo-specific ───────────────────────────────────────────────────────────
    eps: float = 1e-3
    """MeZO perturbation scale ε. The amount to perturb the model by for each forward pass."""

    batch_size: int = 40
    """Per-direction batch size B: the number of examples each (z, ±ε) pair is scored on.
    Orthogonal to z_batch. Total samples per step are batch_size * z_batch."""

    z_batch: int = 1
    """Number of independent (z, batch) pairs averaged per step (Multi-z MeZO, SPSA/ES).
    The total number of samples run per step is z_batch * batch_size, but due to pytorch
    constraints (materializing z) we cannot batch them all together."""

    shared_batch: bool = False
    """Score every z in a step on ONE shared batch instead of z_batch distinct batches
    (i.e. common random numbers across the population).

    Turn this on for centered_rank fitness shaping (ES), where it is required. Ranking members is only
    meaningful when they saw identical data, else the ranks sort data difficulty rather than
    direction quality and bias the update direction.

    For MeZO/SPSA (z_batch averaging), the per-step estimate is a SUM over directions, and sharing
    a batch adds a data-noise floor ∇L(Φ(θ,xᵢ))²/B that more directions can't average away. For the
    same FLOPs you would get strictly higher variance. It could probably be a data efficiency knob,
    but FLOPs is probably the bottleneck and not data.

    The same noise floor issue applies to ES, but it can be mitigated by increasing the batch size."""

    # ── sub-configs ───────────────────────────────────────────────────────────
    mom:          MomentumConfig       = field(default_factory=MomentumConfig)
    seed_window:  SeedWindowConfig     = field(default_factory=SeedWindowConfig)
    perturbation: PerturbationConfig   = field(default_factory=PerturbationConfig)
    clip:         ClipConfig           = field(default_factory=ClipConfig)
    fitness:      FitnessShapingConfig = field(default_factory=FitnessShapingConfig)
    weight_decay: WeightDecayConfig    = field(default_factory=WeightDecayConfig)


@maybe_torchcompile
def zeropower_via_newtonschulz(G: torch.Tensor) -> torch.Tensor:
    """
    Polar Orthogonalization via hybrid Newton-Schulz iteration (DeepSeek-V4 §2.4).
    Stage 1 (8 iters): rapid convergence to the polar factor.
    Stage 2 (2 iters): stabilise singular values precisely at 1.
    """
    assert G.ndim >= 2
    X = G.to(torch.bfloat16)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)

    def _step(X: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
        A = X @ X.T
        return a * X + b * (A @ X) + c * (A @ A @ X)

    for _ in range(8):
        X = _step(X, 3.4445, -4.7750, 2.0315)
    for _ in range(2):
        X = _step(X, 2.0, -1.5, 0.5)

    if transposed:
        X = X.T
    return X.to(G.dtype)


def _aggregate_metrics(dicts: list[dict]) -> dict[str, float]:
    """Average each metric key across all collected (pair, sign) score dicts.
    Tensors are pulled to float here — off the forward-pass critical path."""
    total: dict[str, float] = {}
    count: dict[str, int]   = {}
    for d in dicts:
        for k, v in d.items():
            if v is None:
                continue
            total[k] = total.get(k, 0.0) + (v.item() if isinstance(v, torch.Tensor) else float(v))
            count[k] = count.get(k, 0) + 1
    return {k: total[k] / count[k] for k in total}


class ZOOptimizer:
    def __init__(self, named_params: list, cfg: ZOConfig, score_fn, total_steps: int = 0):
        self.named_params   = named_params
        self.params         = [p.data for _, p in named_params]
        self.cfg            = cfg
        self.score_fn       = score_fn
        self._t             = 0
        self._total_steps   = total_steps
        self.has_momentum   = cfg.mom.momentum_method != "none"
        self._seed_ctr = 0

        # Rolling window of recent |proj_grad| for the adaptive quantile clip threshold.
        self._pg_hist = collections.deque(maxlen=cfg.clip.window) \
            if cfg.clip.strategy != "none" and cfg.clip.clip_pct > 0 else None

        if cfg.mom.momentum_method == "stored_ema":
            self.m = [torch.zeros_like(p) for _, p in named_params]
        elif cfg.mom.momentum_method == "seed_window":
            sw = cfg.seed_window
            buf_cap = sw.size if sw.size > 0 else None
            self.seed_buf = collections.deque(maxlen=buf_cap)
            self.proj_buf = collections.deque(maxlen=buf_cap)

        if cfg.mom.second_moment:
            self.v = [torch.zeros_like(p) for _, p in named_params]

        if cfg.perturbation.distribution != "gaussian":
            assert cfg.mom.momentum_method != "seed_window", \
                "seed_window momentum is incompatible with non-gaussian perturbation distributions"

        self._wd_enabled = cfg.weight_decay.strategy != "none" and cfg.weight_decay.lambda_ > 0
        if cfg.weight_decay.strategy == "anchored" and self._wd_enabled:
            self.theta_0 = [p.data.clone().cpu().pin_memory() for _, p in named_params]

        self.shaping = cfg.fitness.strategy != "none"
        if self.shaping:
            assert cfg.shared_batch, \
                f"{cfg.fitness.strategy} fitness shaping requires shared_batch=True, because the " \
                "population transform is only comparable when all z are scored on the same batch."
            assert not (cfg.clip.strategy != "none" and cfg.clip.clip_pct > 0), \
                f"{cfg.fitness.strategy} fitness shaping is mutually exclusive with proj_grad clipping."


    # ── perturbation ──────────────────────────────────────────────────────────

    def _draw_seeds(self, n: int) -> list[int]:
        """Draw n fresh seeds from a counter that is independent of the training-step
        counter (self._t). A step draws every seed it needs in one call up front, so the
        seed→pair mapping is explicit and fully storable for seed_window reconstruction."""
        seeds = list(range(self._seed_ctr, self._seed_ctr + n))
        self._seed_ctr += n
        return seeds

    def _sample_z(self, seed: int) -> list[torch.Tensor]:
        result = []
        dist     = self.cfg.perturbation.distribution
        rank     = self.cfg.perturbation.rank
        n_params = len(self.named_params)
        for i, (_, p) in enumerate(self.named_params):
            # Derive a seed for each param buffer
            param_seed = (seed * n_params + i) & 0x7FFFFFFF
            z = _philox_randn(param_seed, p.numel(), p.device).view(p.shape)
            if p.ndim >= 2 and dist != "gaussian":
                m, r   = p.shape[0], min(rank, p.shape[0])
                # Both factors MUST be seeded by `seed` so _sample_z is a pure function of it:
                # seed-based regeneration (seed_window momentum, centered_rank's two-pass grad)
                # replays z from the seed and relies on getting the identical perturbation back.
                p_seed = (seed * n_params + i + 2 * n_params) & 0x7FFFFFFF
                P      = _philox_randn(p_seed, m * r, p.device).view(m, r).to(p.dtype)
                if dist == "polar":
                    P, _ = torch.linalg.qr(P)
                psi_seed = (seed * n_params + i + n_params) & 0x7FFFFFFF
                Psi  = _philox_randn(psi_seed, r * p.shape[1], p.device).view(r, p.shape[1]).to(p.dtype)
                if dist == "polar":
                    Psi = zeropower_via_newtonschulz(Psi)
                z = P @ Psi
                # Normalize to ‖z‖_F = √(mn) so eps and lr are invariant to distribution choice.
                # Polar gives ‖z‖_F = √r; low_rank gives √(mnr); Gaussian gives √(mn) in expectation.
                z = z * (p.numel() ** 0.5 / (z.norm() + 1e-7))
            result.append(z)
        return result

    def _get_gradient_ema(self) -> list:
        """Return the first-moment estimate (the update direction): a weighted average
        of past gradient estimates with total weight 1, so it carries gradient magnitude on
        the same scale as a single estimate (matching plain MeZO, so lr transfers)."""
        mc = self.cfg.mom
        if self.cfg.mom.momentum_method == "stored_ema":
            bc = 1.0 - mc.beta1 ** self._t
            return [m / (bc + mc.denom_eps) for m in self.m]
        else:
            return self._reconstruct_momentum_from_seeds()

    def _reconstruct_momentum_from_seeds(self) -> list:
        """Rebuild the first-moment EMA from the seed window"""
        beta1   = self.cfg.mom.beta1
        weights = [beta1 ** i for i in range(len(self.seed_buf))]
        total   = sum(weights) + 1e-12

        momentum = [torch.zeros_like(p) for _, p in self.named_params]
        for step_seeds, step_proj_grads, step_weight in zip(self.seed_buf, self.proj_buf, weights):
            w = step_weight / (len(step_seeds) * total)
            for seed, proj_grad in zip(step_seeds, step_proj_grads):
                torch._foreach_add_(momentum, self._sample_z(seed), alpha=w * proj_grad)
        return momentum

    # ── forward passes ────────────────────────────────────────────────────────

    def _apply_z(self, z: list, scale: float):
        torch._foreach_add_(self.params, z, alpha=scale)

    def run_forward_passes(self, model, batch, z: list, score_fn) -> tuple[tuple[torch.Tensor, dict],
                                                                           tuple[torch.Tensor, dict]]:
        """Run the two perturbed forward passes (0-dim tensors, no device sync).

        Returns ((pos_value, pos_metrics), (neg_value, neg_metrics)). value is the
        scalar to minimize; metrics is the score's breakdown dict (tensors kept lazy
        so the two passes don't sync between them)."""
        def acquire() -> tuple[torch.Tensor, dict]:
            s = score_fn(model, batch)
            value = s.value if isinstance(s.value, torch.Tensor) else torch.tensor(float(s.value))
            return value, s.metrics

        with torch.inference_mode():
            eps = self.cfg.eps
            self._apply_z(z, +eps)
            pos = acquire()
            self._apply_z(z, -2 * eps)
            neg = acquire()
            self._apply_z(z, +eps)
            return pos, neg

    # ── main step ─────────────────────────────────────────────────────────────

    @maybe_torchcompile
    def _apply_update(self, grad: list, lr: torch.Tensor):
        """Step along grad (the first moment / gradient estimate; the projected-gradient
        scalar is already folded in). With second_moment on, divide by the bias-corrected
        √v per dim — the Adam/RMSProp denominator.

        Also applies weight decay according to the configured strategy.
        The AWD paper applies decay after the update, rather than before as standard in Adam, etc.
        Does this matter? I don't know, probably. But the paper doesn't say anything
        about it, so who knows.
        """
        mc     = self.cfg.mom
        params = self.params
        wd     = self.cfg.weight_decay
        lam    = wd.lambda_ * lr

        if mc.second_moment:
            bc2   = 1.0 - mc.beta2 ** self._t
            v_hat = torch._foreach_div(self.v, bc2)
            torch._foreach_sqrt_(v_hat)
            torch._foreach_add_(v_hat, mc.denom_eps)
            torch._foreach_addcdiv_(params, grad, v_hat, value=-lr)  # type: ignore
        else:
            torch._foreach_add_(params, grad, alpha=-lr)  # type: ignore

        if wd.strategy == "standard":
            torch._foreach_mul_(self.params, 1.0 - lam)
        elif wd.strategy == "anchored":
            l1 = wd.penalty == "l1"
            for (_, p), p0 in zip(self.named_params, self.theta_0):
                diff = p.data - p0.to(p.device, non_blocking=True)
                p.data.sub_(torch.sign(diff) if l1 else diff, alpha=lam)

    def _update_optimizer_state(self, grad_est: list,
                                pair_seeds: list[int], pair_grads: list[float]):
        """Update the first-moment backend (stored_ema EMA, or seed_window's seed/scalar
        buffers) and, when enabled, the materialized second moment v (EMA of grad_est²).
        grad_est = (1/Z) Σⱼ clamp(gⱼ)·zⱼ; pair_seeds / pair_grads are this step's per-pair
        seeds and clamped scalars, stored verbatim so seed_window can replay each z."""
        cfg = self.cfg
        mc  = cfg.mom
        if cfg.mom.momentum_method == "stored_ema":
            torch._foreach_mul_(self.m, mc.beta1)
            torch._foreach_add_(self.m, grad_est, alpha=(1 - mc.beta1))
        elif cfg.mom.momentum_method == "seed_window":
            self.seed_buf.appendleft(pair_seeds)
            self.proj_buf.appendleft(pair_grads)
        if mc.second_moment:
            torch._foreach_mul_(self.v, mc.beta2)
            torch._foreach_addcmul_(self.v, grad_est, grad_est, value=1 - mc.beta2)

    def _clip_threshold(self) -> float:
        """Adaptive clip threshold τ: the (1−clip_pct) quantile of recent |proj_grad|.
        Returns +inf when clipping is off or the history window is still empty, so no
        clipping happens during warmup. Built from history only (the current step's own
        scalars are appended afterwards) so a step's outliers can't lift their own bar."""
        cc = self.cfg.clip
        if self._pg_hist is None or not self._pg_hist:
            return float("inf")
        xs  = sorted(self._pg_hist)
        idx = min(len(xs) - 1, int((1.0 - cc.clip_pct) * len(xs)))
        return xs[idx]

    def _apply_grad_clip(self, grad_est: list, pg: list[float], pgc: list[float],
                         tau: float) -> tuple[list[float], float]:
        """Post-loop half of clipping (per_pair already clamped each scalar in the step loop).

        norm strategy: scale the whole grad_est by one factor so the proj_grad vector's
        RMS ≤ τ. Since the zⱼ are near-orthogonal, ‖grad_est‖ ∝ ‖pg‖₂, so this single scalar
        is the exact analog of FO grad-norm clipping — no param-sized reduction. pgc carries
        the same factor so seed_window replay stays exact.

        Also records this step's |proj_grad| into the lagged history window (after τ was
        already computed) and returns (possibly-rescaled pgc, clip_frac for logging)."""
        clip       = self.cfg.clip
        z_batch    = len(pg)
        clip_scale = 1.0
        if clip.strategy == "norm" and math.isfinite(tau):
            pg_norm    = math.sqrt(sum(g * g for g in pg))
            clip_scale = min(1.0, tau * math.sqrt(z_batch) / (pg_norm + 1e-12))
            if clip_scale < 1.0:
                torch._foreach_mul_(grad_est, clip_scale)
                pgc = [g * clip_scale for g in pgc]

        # Append after τ was computed so a step's own outliers can't lift the bar that clips them.
        if self._pg_hist is not None:
            self._pg_hist.extend(abs(g) for g in pg)

        # How much clipping bit this step. per_pair: fraction of pairs clamped at ±τ (high →
        # signSGD regime, τ too tight); norm: 1 if the step's grad_est was scaled down, else 0.
        if clip.strategy == "per_pair" and math.isfinite(tau):
            clip_frac = sum(abs(g) > tau for g in pg) / z_batch
        elif clip.strategy == "norm":
            clip_frac = float(clip_scale < 1.0)
        else:
            clip_frac = 0.0
        return pgc, clip_frac

    @staticmethod
    def _centered_ranks(fitness: list[float]) -> list[float]:
        """Map fitnesses to centered-rank utilities in [-0.5, 0.5] (OpenAI-ES).
        Highest fitness → +0.5, lowest → -0.5; ties broken stably by position. A single
        element maps to 0.0 (no information). This is the rank transform applied to the
        whole step's population before forming the ES gradient."""
        n = len(fitness)
        if n < 2:
            return [0.0] * n
        order = sorted(range(n), key=lambda i: fitness[i])   # ascending: worst first
        ranks = [0.0] * n
        for r, i in enumerate(order):
            ranks[i] = r / (n - 1) - 0.5
        return ranks

    def _shape_fitness(self, vals: list[float]) -> list[float]:
        """Turn the step's 2·z_batch perturbation values into one ES scalar per (z) pair.

        vals is interleaved [val_pos₀, val_neg₀, val_pos₁, val_neg₁, …]; value is minimized, so
        fitness = -value. After the centered-rank transform u, pair j's scalar is (u₋ⱼ - u₊ⱼ):
        the net coefficient of zⱼ from its ±z members, signed so the resulting grad_est points
        uphill in loss (matching proj_grad, so the same θ -= lr·grad_est descends)."""
        utils = self._centered_ranks([-v for v in vals])
        return [utils[2 * j + 1] - utils[2 * j] for j in range(len(vals) // 2)]

    def _grpo_advantages(self, vals: list[float]) -> list[float]:
        """GRPO-style group-relative advantage shaping"""
        n     = len(vals)
        mean  = sum(vals) / n
        adv   = [mean - v for v in vals]
        if self.cfg.fitness.normalize_std:
            var = sum(a * a for a in adv) / n
            std = math.sqrt(var)
            adv = [a / (std + 1e-8) for a in adv]
        return [adv[2 * j + 1] - adv[2 * j] for j in range(n // 2)]

    def step(self, model, batches: list[dict]) -> dict:
        self._t += 1
        cfg       = self.cfg
        z_batch   = cfg.z_batch
        clip      = cfg.clip
        # Per-step threshold from the lagged history. per_pair clamps each scalar to ±τ
        # inside the loop; norm scales the whole grad_est after the loop (needs ‖pg‖₂).
        tau          = self._clip_threshold()
        clip_in_loop = clip.strategy == "per_pair" and math.isfinite(tau)

        lr   = wsd_value(self._t, self._total_steps, lo=0.0, hi=cfg.base.lr, cfg=cfg.base.lr_wsd)
        lr_t = torch.tensor(lr, device=self.named_params[0][1].device)

        # Draw one seed per (z, batch) pair up front (independent of the step counter _t).
        # Stored verbatim so seed_window can later replay each pair's z with its own scalar.
        pair_seeds = self._draw_seeds(z_batch)

        grad_est: list[torch.Tensor] | None = None

        # Per-pair 0-dim tensors collected without syncing; stacked once after the loop.
        val_t        = []        # score.value for each (pair, sign) — the optimization scalar
        pg_t, pgc_t  = [], []    # proj_grad and proj_grad_clipped for each z batch
        metric_dicts = []        # score.metrics for each (pair, sign) — averaged below for logging

        for j, batch in enumerate(batches):
            # `z` here is the MeZO perturbation vector (the direction we probe) — unrelated
            # to any "z_loss" softmax-normalizer term the objective may report in metrics.
            z = self._sample_z(pair_seeds[j])

            # Run the perturbed model and acquire the scalar value (+ metrics) at ±eps.
            (val_pos, m_pos), (val_neg, m_neg) = self.run_forward_passes(model, batch, z, self.score_fn)

            # Calculate the projected grad in the direction of z. per_pair clamps here;
            # norm/none accumulate the raw scalar and (for norm) rescale grad_est later.
            proj_grad         = (val_pos - val_neg) / (2 * cfg.eps)
            proj_grad_clipped = proj_grad.clamp(-tau, tau) if clip_in_loop else proj_grad

            # Accumulate the proj_grad contribution into the gradient estimate. Skipped under
            # fitness shaping: the per-pair scalar isn't known until the whole population has
            # been ranked, so grad_est is built in a second pass below (regenerating each z).
            if not self.shaping:
                if z_batch == 1:
                    grad_est = list(torch._foreach_mul(z, proj_grad_clipped))
                else:
                    if grad_est is None:
                        grad_est = [torch.zeros_like(p) for _, p in self.named_params]
                    torch._foreach_add_(grad_est, z, alpha=proj_grad_clipped / z_batch)  # type: ignore

            # Record metrics for the batch
            val_t.append(val_pos); val_t.append(val_neg)
            pg_t.append(proj_grad); pgc_t.append(proj_grad_clipped)
            metric_dicts.append(m_pos); metric_dicts.append(m_neg)

        # Pull metrics from gpu to python lists
        pg:   list[float] = torch.stack(pg_t).tolist()
        vals: list[float] = torch.stack(val_t).tolist()
        mean_loss: float  = sum(vals) / len(vals)

        if self.shaping:
            # ES/GRPO path: transform the whole population's fitness, then build grad_est by
            # regenerating each zⱼ from its seed and weighting by its shaped scalar (the z's
            # themselves were not kept — too large to hold a full population at once). pgc carries
            # the shaped scalars so seed_window replay reconstructs the same gradient; clipping is off.
            pgc       = (self._grpo_advantages(vals) if cfg.fitness.strategy == "grpo"
                         else self._shape_fitness(vals))
            grad_est  = [torch.zeros_like(p) for _, p in self.named_params]
            for j, seed in enumerate(pair_seeds):
                torch._foreach_add_(grad_est, self._sample_z(seed), alpha=pgc[j] / z_batch)
            clip_frac = 0.0
        else:
            pgc = torch.stack(pgc_t).tolist()
            assert grad_est is not None
            # Apply norm-strategy scaling (no-op for per_pair/none), record the lagged history
            # window, and report how hard clipping bit this step.
            pgc, clip_frac = self._apply_grad_clip(grad_est, pg, pgc, tau)

        # Update optimizer state (first-moment buffers + second moment), then step.
        # The first moment is the EMA when momentum is on, else the raw estimate (so
        # none+second_moment = RMSProp); _apply_update folds in the √v divide if enabled.
        self._update_optimizer_state(grad_est, pair_seeds, pgc)
        m_t = self._get_gradient_ema() if self.has_momentum else grad_est
        self._apply_update(m_t, lr_t)

        # Aggregate metrics. "loss" is the optimization scalar (value, centered over ±eps);
        # the per-criterion breakdown (ce, z_loss, acc, … or name-prefixed for a mixture)
        # is averaged across all (pair, sign) score dicts.
        metrics = {
            "loss":              mean_loss,
            "proj_grad":         sum(pg)  / z_batch,
            "proj_grad_clipped": sum(pgc) / z_batch,
            "clip_frac":         clip_frac,
            "clip_tau":          tau if math.isfinite(tau) else None,
            "lr":                lr,
            **({"fit_rms": (sum(s * s for s in pgc) / z_batch) ** 0.5} if self.shaping else {}),
            **_aggregate_metrics(metric_dicts),
        }
        return metrics


def train_zo(model, tokenizer, total_steps, seed, merge_fn, logger, cfg: ZOConfig | None = None, objective=None, profiling_cfg: ProfilingConfig | None = None):
    """
    Zeroth-order training loop. Perturbs only params with requires_grad=True.
    If merge_fn is provided, calls merge_fn(model) after each step.
    cfg defaults to ZOConfig() (standard MeZO) if not supplied.
    """
    if cfg is None:
        cfg = ZOConfig()
    if objective is None:
        from objective import make_objective
        objective = make_objective("scijudge")

    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    n_params     = sum(p.numel() for _, p in named_params)
    device       = next(model.parameters()).device
    move_fn      = None if getattr(objective, "flat_batches", True) else objective.to_device
    loader       = PrefetchLoader(objective.train_batches(tokenizer, seed, cfg.batch_size), device, move_fn=move_fn)
    run_dir      = os.environ.get("MLSWEEP_RUN_DIR", ".")

    opt = ZOOptimizer(named_params, cfg, objective.score, total_steps=total_steps)
    print_config(cfg)

    print(f"  ZO train: {total_steps:,} steps | {n_params:,} params")
    model.eval()
    show_lr    = not wsd_is_constant(cfg.base.lr_wsd)
    overfit    = cfg.base.overfit_first_batch
    if overfit:
        print("  overfit: reusing the first batch every step")
    fixed_batches = None

    with maybe_enable_profiling(profiling_cfg or ProfilingConfig(), run_dir=run_dir) as torch_profiler:
        for step in range(total_steps):
            if fixed_batches is not None:
                batches = fixed_batches
            elif cfg.shared_batch or overfit:
                # For ES/GRPO Fitness Shaping. Scores the whole z-population on one batch.
                one = get_batches(loader, 1)
                batches = one * cfg.z_batch if one is not None else None
            else:
                # For MeZO/SPSA
                batches = get_batches(loader, cfg.z_batch)
            if batches is None:
                print("Dataloader exhausted, ending training.")
                break
            if overfit:
                fixed_batches = batches

            t0 = time.perf_counter()
            metrics = opt.step(model, batches)
            if merge_fn is not None:
                merge_fn(model)
            metrics["step_time"] = time.perf_counter() - t0

            logger.log(metrics)
            print_zo_step(step, total_steps, metrics, show_lr=show_lr)

            if torch_profiler:
                torch_profiler.step()
                if profiling.trace_saved:
                    print(f"Saved torch profiler trace.")
                    break

            if (step + 1) % cfg.base.ckpt_every == 0:
                save_merged_checkpoint(model, tokenizer, run_dir)
                logger.sync()

    return model

from functools import lru_cache
import torch

from .schedule import maybe_torchcompile


def get_xentropy(fused: bool, compile_mode: str | None = None):
    """Cross-entropy (+ optional z_loss and token accuracy) over the model's LM head,
    returning (loss, z_loss, extra). Selects the fused Liger kernel or the reference
    torch path; the heavy Liger import only happens on the fused path, so eval-only
    callers that never score never pull it in."""
    if fused:
        return _liger_xentropy  # never compiled: graph-breaks
    return _compiled_torch_xentropy(compile_mode)


def _liger_xentropy(model, batch, z_loss_weight: float, compute_accuracy: bool):
    """Fused linear cross-entropy via Liger Kernel.
    When compute_accuracy=True, extra includes 'acc' from Liger's built-in
    return_token_accuracy (computed inside the fused kernel)."""
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

    has_z = z_loss_weight != 0.0
    lce   = LigerFusedLinearCrossEntropyLoss(
        ignore_index=-100,
        lse_square_scale=z_loss_weight,   # λ on mean(logsumexp(logits)²), same as the torch path
        return_z_loss=has_z,
        return_token_accuracy=compute_accuracy,
    )
    m       = getattr(model, "_orig_mod", model)
    hidden  = m.get_decoder()(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
    ).last_hidden_state
    head    = m.get_output_embeddings()
    # Causal shift: hidden[:, :-1] predicts labels[:, 1:], -100 positions ignored.
    hdim    = hidden.size(-1)
    shift_h = hidden[:, :-1, :].reshape(-1, hdim)
    shift_t = batch["labels"][:, 1:].reshape(-1)
    out     = lce(head.weight, shift_h, shift_t, getattr(head, "bias", None))

    if isinstance(out, torch.Tensor):
        # Neither return_z_loss nor return_token_accuracy; plain scalar tensor.
        return out, None, None

    # CrossEntropyOutput — at least one extra return flag is active.
    extra = {"acc": out.token_accuracy} if compute_accuracy else None
    if not has_z:
        return out.loss, None, extra
    # Liger folds z_loss INTO out.loss; our contract keeps CE and z_loss distinct
    # so the train loops can log both and sum them.
    return out.loss - out.z_loss, out.z_loss, extra


def _torch_xentropy(model, batch, z_loss_weight: float, compute_accuracy: bool):
    """Reference path: full forward, cross-entropy from materialized logits.
    Returns (loss, z_loss, extra) where extra is {"acc": tensor} or None."""
    out  = model(**batch)
    loss = out.loss

    extra = None
    if compute_accuracy:
        # Causal shift: logits[:, :-1] predict labels[:, 1:], -100 ignored.
        # Masked arithmetic (no boolean indexing) for torch.compile fullgraph compat.
        logits = out.logits[:, :-1, :]
        labels = batch["labels"][:, 1:]
        mask   = labels != -100
        correct = (logits.argmax(-1) == labels) & mask
        extra   = {"acc": correct.sum() / mask.sum().clamp_min(1)}

    if z_loss_weight == 0.0:
        return loss, None, extra

    logits = out.logits[:, :-1, :].float()
    labels = batch["labels"][:, 1:]
    mask   = labels != -100
    lse    = torch.logsumexp(logits, dim=-1)  # log Z per position
    z_loss = z_loss_weight * (lse.square() * mask).sum() / mask.sum().clamp_min(1)
    return loss, z_loss, extra


@lru_cache(maxsize=None)
def _compiled_torch_xentropy(compile_mode: str | None):
    """The reference path, compiled once per (enabled, mode). torch.compile must reuse
    one artifact — rebuilding it per step would re-trace and defeat compilation — so
    the build-once cache lives here in the kernel module, not on the criterion."""
    return maybe_torchcompile(_torch_xentropy, enabled=(compile_mode is None), mode=compile_mode)

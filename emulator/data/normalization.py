"""Feature and target normalization utilities.

The normalization path is deliberately shared by training and inference so
checkpointed statistics are applied consistently.
"""

from __future__ import annotations

import torch


def _as_feature_vector(t: torch.Tensor, name: str) -> torch.Tensor:
    """Ensure a stats tensor is reduced to shape `(F,)`.

    Older checkpoints may store unexpectedly large tensors (e.g., per-sample
    arrays). To prevent GPU OOM during inference, all leading dimensions are
    reduced by averaging while preserving the final feature dimension.
    """
    if not torch.is_tensor(t):
        t = torch.as_tensor(t)
    t = t.detach().to(dtype=torch.float32, device="cpu")

    if t.ndim == 1:
        return t
    if t.ndim == 0:
        raise ValueError(f"{name} is a scalar; expected shape (F,)")

    feature_dim = t.shape[-1]
    n = int(t.numel() // feature_dim)
    reduced = t.reshape(n, feature_dim).mean(dim=0)

    print(
        f"[WARN] {name} had shape {tuple(t.shape)}; reduced to (F,)={tuple(reduced.shape)} to avoid GPU OOM.",
        flush=True,
    )
    return reduced


def normalize_inputs_inplace(
    batch,
    x_center: torch.Tensor,
    x_scale: torch.Tensor,
    x_clip: float = 5.0,
    do_aug: bool = False,
    aug_prob: float = 1.0,
    aug_scale: float = 0.05,
    aug_bias: float = 0.02,
    pmean_center: torch.Tensor | None = None,
    pmean_scale: torch.Tensor | None = None,
    pmean_clip: float | None = None,
):
    """Normalize `batch.x` / `batch.x_hist` (and optional p_mean metadata) in place.

    During training, optional feature jitter can be enabled in normalized space.
    In inference, callers keep `do_aug=False`.
    """
    x_clip = float(x_clip)
    aug_prob = float(aug_prob)
    aug_scale = float(aug_scale)
    aug_bias = float(aug_bias)

    # 1) Node feature normalization.
    batch.x = batch.x.float()
    batch.x.sub_(x_center).div_(x_scale)
    if x_clip > 0:
        batch.x.clamp_(-x_clip, x_clip)

    if hasattr(batch, "x_hist"):
        batch.x_hist = batch.x_hist.float()
        batch.x_hist.sub_(x_center.view(1, 1, -1)).div_(x_scale.view(1, 1, -1))
        if x_clip > 0:
            batch.x_hist.clamp_(-x_clip, x_clip)

    # 2) Optional global mean pressure normalization.
    if (pmean_center is not None) and (pmean_scale is not None):
        clip_value = float(x_clip if pmean_clip is None else pmean_clip)

        if hasattr(batch, "p_mean_hist"):
            p_mean_hist = batch.p_mean_hist.float()
            p_mean_hist = (p_mean_hist - pmean_center) / pmean_scale
            if clip_value > 0:
                p_mean_hist = p_mean_hist.clamp(-clip_value, clip_value)
            batch.p_mean_hist = p_mean_hist

        if hasattr(batch, "p_mean_curr"):
            p_mean_curr = batch.p_mean_curr.float().view(-1)
            p_mean_curr = (p_mean_curr - pmean_center) / pmean_scale
            if clip_value > 0:
                p_mean_curr = p_mean_curr.clamp(-clip_value, clip_value)
            batch.p_mean_curr = p_mean_curr

    # 3) Optional train-time augmentation in normalized space.
    if do_aug:
        if (aug_prob >= 1.0) or (torch.rand((), device=batch.x.device) < aug_prob):
            feature_dim = batch.x.size(-1)
            scale = (1.0 + (2.0 * torch.rand((feature_dim,), device=batch.x.device) - 1.0) * aug_scale).view(1, -1)
            bias = (torch.randn((feature_dim,), device=batch.x.device) * aug_bias).view(1, -1)

            batch.x.mul_(scale).add_(bias)
            if x_clip > 0:
                batch.x.clamp_(-x_clip, x_clip)

            if hasattr(batch, "x_hist"):
                batch.x_hist.mul_(scale.view(1, 1, -1)).add_(bias.view(1, 1, -1))
                if x_clip > 0:
                    batch.x_hist.clamp_(-x_clip, x_clip)


def normalize_targets(y_raw, y_mean, y_std):
    """Normalize targets from physical units to standardized units."""
    return (y_raw - y_mean) / y_std


__all__ = [
    "_as_feature_vector",
    "normalize_inputs_inplace",
    "normalize_targets",
]

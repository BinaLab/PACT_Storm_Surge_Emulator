"""Loss functions used by training.

All losses operate in physical target units to preserve interpretability and
stable checkpoint selection.
"""

from __future__ import annotations

import torch


def weighted_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    q_value: torch.Tensor,
    alpha: float,
    s: float,
    use_abs: bool = True,
):
    """Weighted MSE emphasizing large-magnitude targets.

    Weight definition:
      w(y) = 1 + alpha * sigmoid((|y| - q) / s)
    """
    alpha = float(alpha)
    s = float(max(s, 1e-6))

    yy = target.abs() if use_abs else target
    w = 1.0 + alpha * torch.sigmoid((yy - q_value) / s)
    return (w * (pred - target).pow(2)).mean()


def _robust_charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Elementwise Charbonnier penalty: sqrt(x^2 + eps^2)."""
    eps = float(max(eps, 1e-12))
    return torch.sqrt(x * x + (eps * eps))


def _robust_huber(x: torch.Tensor, delta: float = 0.05) -> torch.Tensor:
    """Elementwise Huber penalty."""
    delta = float(max(delta, 1e-12))
    ax = x.abs()
    quad = 0.5 * (x * x)
    lin = delta * (ax - 0.5 * delta)
    return torch.where(ax <= delta, quad, lin)


def slope_matching_loss_softmask(
    pred: torch.Tensor,
    target: torch.Tensor,
    tau: torch.Tensor,
    mask_s: float = 0.10,
    robust: str = "charb",
    charb_eps: float = 1e-3,
    huber_delta: float = 0.05,
) -> torch.Tensor:
    """First-difference matching along forecast horizon with a soft peak mask."""
    if pred.ndim != 2 or target.ndim != 2:
        raise ValueError(
            "slope_matching_loss_softmask expects (B,H) tensors, "
            f"got {pred.shape} and {target.shape}"
        )
    if pred.size(1) < 2:
        return pred.new_zeros(())

    mask_s = float(max(mask_s, 1e-6))

    dp = pred[:, 1:] - pred[:, :-1]
    dy = target[:, 1:] - target[:, :-1]
    err = dp - dy

    if robust == "huber":
        per = _robust_huber(err, delta=huber_delta)
    else:
        per = _robust_charbonnier(err, eps=charb_eps)

    peak_abs = target.abs().max(dim=1).values
    w = torch.sigmoid((tau - peak_abs) / mask_s)
    per = per * w.unsqueeze(1)
    return per.mean()


__all__ = [
    "weighted_mse_loss",
    "slope_matching_loss_softmask",
]

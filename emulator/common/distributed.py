"""Single-node distributed helpers.

These wrappers keep DDP usage uniform across the project and make it safe to run
in both distributed and single-process modes.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def ddp_is_initialized() -> bool:
    """Return True when torch.distributed is ready to communicate."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Return the global rank (0 in non-DDP runs)."""
    return dist.get_rank() if ddp_is_initialized() else 0


def get_world_size() -> int:
    """Return the world size (1 in non-DDP runs)."""
    return dist.get_world_size() if ddp_is_initialized() else 1


def is_main_process() -> bool:
    """True on rank 0."""
    return get_rank() == 0


def print0(*args, **kwargs) -> None:
    """Rank-0 print helper to avoid duplicated logs in DDP."""
    if is_main_process():
        print(*args, **kwargs)


def ddp_all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce a tensor with SUM when DDP is initialized."""
    if ddp_is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


__all__ = [
    "ddp_is_initialized",
    "get_rank",
    "get_world_size",
    "is_main_process",
    "print0",
    "ddp_all_reduce_sum",
]

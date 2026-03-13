"""Runtime helpers for deterministic experiments and CPU threading control."""

from __future__ import annotations

import os
import random
from contextlib import contextmanager

import numpy as np
import torch


def infer_dataset_tag(root_dir: str) -> str:
    """Infer a compact dataset identifier from a root path.

    Preferred pattern:
      ./Data/<DATASET>/graphs

    Fallbacks:
    - if path ends with `graphs`, use the parent folder name.
    - otherwise use the leaf folder name.
    """
    if root_dir is None:
        return "data"

    path = os.path.normpath(str(root_dir))
    parts = path.split(os.sep)

    # Scan for .../Data/<name>/graphs
    for i in range(len(parts) - 2):
        if parts[i].lower() == "data" and parts[i + 2].lower() == "graphs":
            name = parts[i + 1].strip()
            return name if name else "data"

    if len(parts) >= 2 and parts[-1].lower() == "graphs":
        name = parts[-2].strip()
        return name if name else "data"

    leaf = os.path.basename(path).strip()
    return leaf if leaf else "data"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def temp_numpy_threads(num_threads: int):
    """Temporarily override OMP/MKL thread counts for one-time NumPy work.

    This is used for expensive percentile/statistics computation without affecting
    the rest of the training process.
    """
    num_threads = int(max(1, num_threads))
    old_omp = os.environ.get("OMP_NUM_THREADS")
    old_mkl = os.environ.get("MKL_NUM_THREADS")

    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    try:
        yield
    finally:
        if old_omp is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = old_omp

        if old_mkl is None:
            os.environ.pop("MKL_NUM_THREADS", None)
        else:
            os.environ["MKL_NUM_THREADS"] = old_mkl


def infer_stats_threads(torch_threads: int) -> int:
    """Choose a practical CPU thread count for one-time NumPy statistics."""
    tt = int(max(1, torch_threads))
    if tt > 1:
        return tt

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
    if slurm_cpus.isdigit():
        n = int(slurm_cpus)
    else:
        n = os.cpu_count() or 16

    return max(4, min(n, 32))


__all__ = [
    "infer_dataset_tag",
    "set_seed",
    "temp_numpy_threads",
    "infer_stats_threads",
]

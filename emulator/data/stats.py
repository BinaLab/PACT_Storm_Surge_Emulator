"""Training-data statistics for normalization and loss shaping.

All statistics are computed on the training split only, and distributed-safe
paths are provided for DDP runs.
"""

from __future__ import annotations

import numpy as np
import torch

from emulator.common.distributed import ddp_all_reduce_sum, get_rank, get_world_size
from emulator.data.graph_store import ForcingGraphStore


def compute_x_stats_distributed_from_store(store: ForcingGraphStore, train_indices, device):
    """Compute distributed feature mean/std from `g.x` over training graphs."""
    rank = get_rank()
    world = get_world_size()

    local_indices = train_indices[rank::world]
    feat_dim = store.graphs[train_indices[0]].x.size(-1)

    sum_x = torch.zeros(feat_dim, device=device, dtype=torch.float64)
    sum_x2 = torch.zeros(feat_dim, device=device, dtype=torch.float64)
    count = torch.zeros((), device=device, dtype=torch.float64)

    for idx in local_indices:
        x = store.graphs[idx].x.to(device=device, dtype=torch.float32, non_blocking=True)
        sum_x += x.sum(dim=0, dtype=torch.float64)
        sum_x2 += (x * x).sum(dim=0, dtype=torch.float64)
        count += x.size(0)

    ddp_all_reduce_sum(sum_x)
    ddp_all_reduce_sum(sum_x2)
    ddp_all_reduce_sum(count)

    mean = (sum_x / count).to(torch.float32)
    var = (sum_x2 / count).to(torch.float32) - mean**2
    std = torch.sqrt(var + 1e-6)
    return mean, std


def compute_x_robust_stats_from_store_rank0(
    store: ForcingGraphStore,
    train_indices,
    p_lo: float = 1.0,
    p_hi: float = 99.0,
    nodes_per_graph: int = 256,
    seed: int = 0,
):
    """Compute robust per-feature center/scale from sampled training nodes."""
    rng = np.random.default_rng(int(seed))
    nodes_per_graph = int(max(1, nodes_per_graph))
    p_lo = float(p_lo)
    p_hi = float(p_hi)
    assert 0.0 <= p_lo < p_hi <= 100.0

    g0 = store.graphs[train_indices[0]]
    feat_dim = int(g0.x.size(-1))

    total_rows = 0
    for idx in train_indices:
        x = store.graphs[idx].x
        if x is None:
            continue
        n = int(x.size(0))
        total_rows += min(n, nodes_per_graph)

    if total_rows <= 0:
        p_lo_vec = np.zeros(feat_dim, dtype=np.float32)
        p_hi_vec = np.ones(feat_dim, dtype=np.float32)
    else:
        x_all = np.empty((total_rows, feat_dim), dtype=np.float32)
        pos = 0

        for idx in train_indices:
            x_np = store.graphs[idx].x.detach().cpu().numpy()
            if x_np.ndim != 2:
                continue
            n = x_np.shape[0]
            k = min(n, nodes_per_graph)
            if n <= nodes_per_graph:
                x_all[pos : pos + k] = x_np.astype(np.float32, copy=False)
            else:
                sel = rng.choice(n, size=k, replace=False)
                x_all[pos : pos + k] = x_np[sel].astype(np.float32, copy=False)
            pos += k

        q = np.percentile(x_all, [p_lo, p_hi], axis=0, overwrite_input=True)
        p_lo_vec = q[0].astype(np.float32, copy=False)
        p_hi_vec = q[1].astype(np.float32, copy=False)

    center = 0.5 * (p_lo_vec + p_hi_vec)
    scale = 0.5 * (p_hi_vec - p_lo_vec)
    scale = np.maximum(scale, 1e-6).astype(np.float32, copy=False)

    return (
        torch.from_numpy(center),
        torch.from_numpy(scale),
        torch.from_numpy(p_lo_vec),
        torch.from_numpy(p_hi_vec),
    )


def compute_x_mag_stats_from_store_rank0(
    store: ForcingGraphStore,
    train_indices,
    p_hi: float = 99.0,
    nodes_per_graph: int = 256,
    seed: int = 0,
):
    """Compute magnitude-only feature scaling (percentile of |x|)."""
    rng = np.random.default_rng(int(seed))
    nodes_per_graph = int(max(1, nodes_per_graph))
    p_hi = float(p_hi)
    assert 0.0 < p_hi <= 100.0

    g0 = store.graphs[train_indices[0]]
    feat_dim = int(g0.x.size(-1))

    total_rows = 0
    for idx in train_indices:
        x = store.graphs[idx].x
        if x is None:
            continue
        n = int(x.size(0))
        total_rows += min(n, nodes_per_graph)

    if total_rows <= 0:
        return torch.ones(feat_dim, dtype=torch.float32)

    x_abs = np.empty((total_rows, feat_dim), dtype=np.float32)
    pos = 0
    for idx in train_indices:
        x_np = store.graphs[idx].x.detach().cpu().numpy()
        if x_np.ndim != 2:
            continue
        n = x_np.shape[0]
        k = min(n, nodes_per_graph)

        if n <= nodes_per_graph:
            np.abs(x_np[:k], out=x_abs[pos : pos + k])
        else:
            sel = rng.choice(n, size=k, replace=False)
            np.abs(x_np[sel], out=x_abs[pos : pos + k])

        pos += k

    magnitude = np.percentile(x_abs, p_hi, axis=0, overwrite_input=True)
    magnitude = np.maximum(magnitude, 1e-6).astype(np.float32, copy=False)
    return torch.from_numpy(magnitude)


def _graph_has_pmean(graph) -> bool:
    """Return True if a graph carries global mean pressure metadata."""
    return hasattr(graph, "p_mean_hist") or hasattr(graph, "p_mean_curr")


def compute_pmean_stats_from_store_rank0(
    store: ForcingGraphStore,
    train_indices,
    history_steps: int,
    mode: str,
    p_lo: float = 1.0,
    p_hi: float = 99.0,
):
    """Compute scalar p_mean normalization stats from the training split.

    Returns:
      - center tensor
      - scale tensor
      - dict with debug metadata (percentiles or magnitude)

    If p_mean metadata is absent in the dataset, returns `(None, None, None)`.
    """
    mode = str(mode)
    if mode not in ("zscore", "robust", "mag"):
        raise ValueError(f"Unsupported p_mean norm mode: {mode}")

    any_has = False
    for idx in train_indices[: min(len(train_indices), 256)]:
        if _graph_has_pmean(store.graphs[idx]):
            any_has = True
            break
    if not any_has:
        return None, None, None

    window = int(history_steps) + 1
    values = []

    for idx in train_indices:
        g = store.graphs[idx]
        if hasattr(g, "p_mean_hist"):
            p_mean_hist = g.p_mean_hist
            if torch.is_tensor(p_mean_hist):
                p_mean_hist = p_mean_hist.detach().cpu()
            if p_mean_hist.ndim == 2 and p_mean_hist.shape[-1] == 1:
                p_mean_hist = p_mean_hist.squeeze(-1)
            p_mean_hist = p_mean_hist.reshape(-1)
            p_mean_hist = p_mean_hist[-window:] if p_mean_hist.numel() >= window else p_mean_hist
            values.append(p_mean_hist.numpy().astype(np.float32, copy=False))
        elif hasattr(g, "p_mean_curr"):
            p_curr = g.p_mean_curr
            if torch.is_tensor(p_curr):
                p_curr = float(p_curr.detach().cpu().view(-1)[0].item())
            else:
                p_curr = float(p_curr)
            values.append(np.array([p_curr], dtype=np.float32))

    if len(values) == 0:
        return None, None, None

    arr = np.concatenate(values, axis=0).astype(np.float32, copy=False)
    extra = {}

    if mode == "zscore":
        mu = float(arr.mean(dtype=np.float64))
        sd = float(arr.std(dtype=np.float64))
        sd = max(sd, 1e-6)
        extra.update(dict(mean=mu, std=sd))
        return torch.tensor(mu, dtype=torch.float32), torch.tensor(sd, dtype=torch.float32), extra

    if mode == "robust":
        p_lo = float(p_lo)
        p_hi = float(p_hi)
        assert 0.0 <= p_lo < p_hi <= 100.0
        q0, q1 = np.percentile(arr, [p_lo, p_hi])
        center = 0.5 * (float(q0) + float(q1))
        scale = 0.5 * (float(q1) - float(q0))
        scale = max(scale, 1e-6)
        extra.update(dict(p_lo=float(q0), p_hi=float(q1)))
        return torch.tensor(center, dtype=torch.float32), torch.tensor(scale, dtype=torch.float32), extra

    p_hi = float(p_hi)
    assert 0.0 < p_hi <= 100.0
    mag = float(np.percentile(np.abs(arr), p_hi))
    mag = max(mag, 1e-6)
    extra.update(dict(mag=mag, mag_p=p_hi))
    return torch.tensor(0.0, dtype=torch.float32), torch.tensor(mag, dtype=torch.float32), extra


def compute_y_stats_distributed_from_store(store: ForcingGraphStore, train_indices, device):
    """Compute distributed target mean/std from `g.y` over training graphs."""
    rank = get_rank()
    world = get_world_size()

    local_indices = train_indices[rank::world]
    out_dim = store.graphs[train_indices[0]].y.numel()

    sum_y = torch.zeros(out_dim, device=device, dtype=torch.float64)
    sum_y2 = torch.zeros(out_dim, device=device, dtype=torch.float64)
    count = torch.zeros((), device=device, dtype=torch.float64)

    for idx in local_indices:
        y = store.graphs[idx].y.view(-1).to(device=device, dtype=torch.float64)
        sum_y += y
        sum_y2 += y**2
        count += 1.0

    ddp_all_reduce_sum(sum_y)
    ddp_all_reduce_sum(sum_y2)
    ddp_all_reduce_sum(count)

    mean = (sum_y / count).to(torch.float32)
    var = (sum_y2 / count).to(torch.float32) - mean**2
    std = torch.sqrt(var + 1e-6)
    return mean, std


def compute_train_loss_thresholds_from_store(
    store: ForcingGraphStore,
    train_indices,
    wmse_q_percentile: float = 95.0,
    tail_frac: float = 0.05,
    wmse_use_abs: bool = True,
):
    """Compute train-derived thresholds for weighted and tail-aware losses."""
    wmse_q_percentile = float(wmse_q_percentile)
    tail_frac = float(tail_frac)
    tail_frac = min(max(tail_frac, 1e-6), 0.999999)

    ys = []
    peaks = []

    for idx in train_indices:
        y = store.graphs[idx].y.detach().cpu().view(-1).float().numpy()
        ys.append(np.abs(y) if wmse_use_abs else y)
        peaks.append(float(y.max()))

    if len(ys) == 0:
        return 0.0, 0.0

    y_all = np.concatenate(ys, axis=0)
    peak_all = np.array(peaks, dtype=np.float32)

    wmse_q_value = float(np.percentile(y_all, wmse_q_percentile))
    tail_peak_thr = float(np.percentile(peak_all, 100.0 * (1.0 - tail_frac)))
    return wmse_q_value, tail_peak_thr


__all__ = [
    "compute_x_stats_distributed_from_store",
    "compute_x_robust_stats_from_store_rank0",
    "compute_x_mag_stats_from_store_rank0",
    "_graph_has_pmean",
    "compute_pmean_stats_from_store_rank0",
    "compute_y_stats_distributed_from_store",
    "compute_train_loss_thresholds_from_store",
]

"""Graph dataset storage and split/view helpers.

This module centralizes all graph file handling so training and inference use the
same filename parsing, station filtering, and year-based splitting logic.
"""

from __future__ import annotations

import glob
import os
import random
from collections import defaultdict
from typing import Callable

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


def _log(log_fn: Callable[..., None], message: str) -> None:
    """Small logging shim so callers can pass `print` or rank-aware `print0`."""
    try:
        log_fn(message, flush=True)
    except TypeError:
        # Some callsites may provide wrappers that do not accept `flush`.
        log_fn(message)


class ForcingGraphStore:
    """Load matching `*_graphs.pt` files once and index them by year/station.

    Filename convention:
      <year0>_<year1>_<station>_<version...>_graphs.pt

    Attributes:
      - graphs: list of `torch_geometric.data.Data`
      - graph_tags: list of string tags aligned with `graphs`
      - year_to_indices: dict mapping `<year0>_<year1>` -> sample indices
      - station_to_indices: dict mapping station key -> sample indices
      - station_filter: effective filename-level station filter, or `None`
    """

    def __init__(
        self,
        root_dir: str,
        pattern: str = "*graphs.pt",
        log_fn: Callable[..., None] = print,
        force_cpu: bool = True,
        station_filter: str | None = None,
        strict_station_filter: bool = True,
    ):
        self.root_dir = root_dir
        self.pattern = pattern
        self.log_fn = log_fn

        self.graphs: list[Data] = []
        self.graph_tags: list[str] = []
        self.year_to_indices: dict[str, list[int]] = defaultdict(list)
        self.station_to_indices: dict[str, list[int]] = defaultdict(list)

        files = sorted(glob.glob(os.path.join(root_dir, pattern)))
        _log(log_fn, f"Found {len(files)} graph files in {root_dir}")

        # Parse every filename before loading any payload. This makes station
        # filtering cheap and prevents unrelated stations from ever reaching
        # torch.load (and therefore from occupying CPU RAM).
        file_records: list[tuple[str, str, str, str]] = []
        for fpath in files:
            base = os.path.basename(fpath)
            if base.endswith("_graphs.pt"):
                stem = base[: -len("_graphs.pt")]
            else:
                stem = os.path.splitext(base)[0]

            parts = stem.split("_")
            if len(parts) < 3:
                raise ValueError(f"Unexpected filename pattern: {stem}")

            year_tag = f"{parts[0]}_{parts[1]}"
            station_tag = parts[2]
            version_tag = "_".join(parts[3:]) if len(parts) > 3 else ""

            file_records.append((fpath, year_tag, station_tag, version_tag))

        requested_station = None if station_filter is None else str(station_filter).strip()
        if requested_station == "":
            requested_station = None

        available_stations = sorted({record[2] for record in file_records})
        if requested_station is not None:
            if requested_station not in available_stations:
                message = (
                    f"Station filter '{requested_station}' not found in filenames under {root_dir}. "
                    f"Available={available_stations}"
                )
                if strict_station_filter:
                    raise RuntimeError(message)
                _log(log_fn, f"[WARN] {message} -> Loading ALL station files.")
                requested_station = None
            else:
                selected_records = [record for record in file_records if record[2] == requested_station]
                _log(
                    log_fn,
                    f"[Store filter] station='{requested_station}' files="
                    f"{len(selected_records)}/{len(file_records)} selected before torch.load.",
                )
                file_records = selected_records

        self.station_filter = requested_station

        for fpath, year_tag, station_tag, version_tag in file_records:
            # Loading to CPU avoids hidden CUDA allocations from serialized tensors.
            if force_cpu:
                data_list = torch.load(fpath, map_location="cpu", weights_only=False)
            else:
                data_list = torch.load(fpath, weights_only=False)

            start_idx = len(self.graphs)
            for i, graph in enumerate(data_list):
                if force_cpu:
                    try:
                        graph = graph.cpu()
                    except Exception:
                        # Keep original object if it is not a standard Data-like class.
                        pass

                self.graphs.append(graph)
                self.graph_tags.append(f"{year_tag}_{station_tag}_{version_tag}_{i}")

                idx = start_idx + i
                self.year_to_indices[year_tag].append(idx)
                self.station_to_indices[station_tag].append(idx)

        _log(log_fn, f"Loaded {len(self.graphs)} graphs from {len(file_records)} files.")
        _log(log_fn, f"Years found: {sorted(self.year_to_indices.keys())}")
        _log(log_fn, f"Stations found: {sorted(self.station_to_indices.keys())}")


def years_from_indices(store: ForcingGraphStore, indices) -> list[str]:
    """Return sorted unique year tags from a list of sample indices."""
    years = set()
    for idx in indices:
        tag = store.graph_tags[idx]
        parts = tag.split("_")
        if len(parts) >= 2:
            years.add(f"{parts[0]}_{parts[1]}")
    return sorted(years)


def _year_tag_has_year_after(year_tag: str, threshold: int) -> bool:
    """Return True when any year component in a tag like `2070_2071` exceeds threshold."""
    try:
        years = [int(part) for part in year_tag.split("_")[:2]]
    except ValueError as exc:
        raise ValueError(f"Could not parse year tag '{year_tag}' for future-only split.") from exc
    return any(year > int(threshold) for year in years)


def make_year_split_indices(
    store: ForcingGraphStore,
    part: str,
    station_filter: str | None,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    shuffle_years: bool = False,
    future_only: bool = False,
    future_year_threshold: int = 2030,
    split_seed: int = 42,
    log_fn: Callable[..., None] = print,
):
    """Create deterministic year-group train/val/test splits.

    Splitting is performed over year groups (not individual samples), which keeps
    temporal leakage controlled. By default, groups are sorted chronologically;
    optionally, year groups can be filtered to future-only tags before slicing,
    or shuffled deterministically before slicing by ratio.
    """
    year_to_indices = store.year_to_indices
    station_to_indices = store.station_to_indices

    if station_filter is not None:
        if station_filter in station_to_indices:
            allowed = set(station_to_indices[station_filter])
            _log(log_fn, f"[Filter] station='{station_filter}' found: {len(allowed)} samples allowed.")
        else:
            _log(log_fn, f"[WARN] station filter '{station_filter}' not found. Using full dataset.")
            allowed = None
            station_filter = None
    else:
        allowed = None

    group_to_indices: dict[str, list[int]] = {}
    for year_tag, idxs in year_to_indices.items():
        filtered = list(idxs) if allowed is None else [i for i in idxs if i in allowed]
        if filtered:
            group_to_indices[year_tag] = filtered

    if future_only:
        n_before = len(group_to_indices)
        group_to_indices = {
            year_tag: idxs
            for year_tag, idxs in group_to_indices.items()
            if _year_tag_has_year_after(year_tag, future_year_threshold)
        }
        _log(
            log_fn,
            f"[Split filter] future_only=1 year>{int(future_year_threshold)} "
            f"kept={len(group_to_indices)}/{n_before} year groups.",
        )

    year_groups = sorted(group_to_indices.keys())
    if shuffle_years:
        rng = random.Random(int(split_seed))
        rng.shuffle(year_groups)

    n = len(year_groups)
    if n == 0:
        raise RuntimeError("No years found after applying station/future split filters.")

    def split_groups(group_list, train_ratio: float, val_ratio: float):
        m = len(group_list)
        if m == 1:
            return group_list, [], []
        if m == 2:
            return [group_list[0]], [group_list[1]], []

        n_train = max(1, int(round(train_ratio * m)))
        n_val = max(1, int(round(val_ratio * m)))
        if n_train + n_val >= m:
            n_train = max(1, m - 2)
            n_val = 1

        train_years = group_list[:n_train]
        val_years = group_list[n_train : n_train + n_val]
        test_years = group_list[n_train + n_val :]
        return train_years, val_years, test_years

    train_years, val_years, test_years = split_groups(year_groups, train_frac, val_frac)
    selected_years = train_years if part == "train" else (val_years if part == "val" else test_years)

    indices = []
    for year_tag in selected_years:
        indices.extend(group_to_indices[year_tag])
    indices = sorted(indices)

    order_note = f"shuffled(seed={int(split_seed)})" if shuffle_years else "sorted"
    if future_only:
        order_note += f",future_year>{int(future_year_threshold)}"
    if station_filter is not None:
        _log(log_fn, f"[Split] part={part} station='{station_filter}' order={order_note} years={selected_years} samples={len(indices)}")
    else:
        _log(log_fn, f"[Split] part={part} order={order_note} years={selected_years} samples={len(indices)}")

    return indices


def make_all_years_test_indices(
    store: ForcingGraphStore,
    station_filter: str | None,
    strict: bool = False,
    log_fn: Callable[..., None] = print,
):
    """Return all test indices, optionally filtered by station."""
    if station_filter is not None:
        if station_filter in store.station_to_indices:
            idx = sorted(store.station_to_indices[station_filter])
            _log(log_fn, f"[External test] station='{station_filter}' FOUND -> samples={len(idx)}")
            return idx

        message = (
            f"[External test] station='{station_filter}' NOT FOUND. "
            f"Available={sorted(store.station_to_indices.keys())}"
        )
        if strict:
            raise RuntimeError(message)
        _log(log_fn, f"[WARN] {message} -> Using ALL samples.")

    return list(range(len(store.graphs)))


class ForcingGraphView(Dataset):
    """Dataset view that slices history windows without mutating stored graphs.

    Returned sample fields:
      - `x`: node features `(N, F)`
      - `edge_index`: graph connectivity
      - `y`: forecast target `(1, H)`
      - `x_hist`: optional history tensor `(N, W, F)`
      - `grid_H` / `grid_W`: optional grid dimensions used by the CNN encoder
      - `p_mean_hist` / `p_mean_curr`: optional global pressure metadata
      - `tag`: string identifier for reproducibility and grouped metrics
    """

    def __init__(self, store: ForcingGraphStore, indices, history_steps: int):
        self.store = store
        self.indices = list(indices)
        self.history_steps = int(history_steps)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        g = self.store.graphs[idx]

        data = Data()
        data.x = g.x
        data.edge_index = g.edge_index
        data.y = g.y.view(1, -1)

        # Keep spatial metadata available after PyG batching. GraphSAGE does not
        # need it; the CNN encoder uses it to recover (B, C, H, W) dynamically.
        if hasattr(g, "grid_H"):
            data.grid_H = int(g.grid_H)
        if hasattr(g, "grid_W"):
            data.grid_W = int(g.grid_W)

        window = self.history_steps + 1
        if hasattr(g, "x_hist"):
            x_hist = g.x_hist[-window:]  # (W, N, F)
            data.x_hist = x_hist.permute(1, 0, 2)  # (N, W, F)
        elif self.history_steps == 0:
            # PACT's 0h control also accepts graphs containing current forcing only.
            data.x_hist = data.x.unsqueeze(1)

        # Pressure metadata also belongs to spatial-only graphs without x_hist.
        if hasattr(g, "p_mean_hist"):
            p_mean_hist = g.p_mean_hist
            if p_mean_hist.dim() == 2 and p_mean_hist.size(-1) == 1:
                p_mean_hist = p_mean_hist.squeeze(-1)

            p_mean_hist = p_mean_hist[-window:]
            data.p_mean_hist = p_mean_hist.view(1, -1)
            data.p_mean_curr = p_mean_hist[-1].view(1)
        elif hasattr(g, "p_mean_curr"):
            if torch.is_tensor(g.p_mean_curr):
                data.p_mean_curr = g.p_mean_curr.view(1)
            else:
                data.p_mean_curr = torch.tensor([g.p_mean_curr], dtype=torch.float32)

        data.tag = self.store.graph_tags[idx]
        return data


__all__ = [
    "ForcingGraphStore",
    "ForcingGraphView",
    "years_from_indices",
    "make_year_split_indices",
    "make_all_years_test_indices",
]

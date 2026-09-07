#!/usr/bin/env python3
# time_align_unified.py
#
# -----------------------------------------------------------------------------
# GOAL
# -----------------------------------------------------------------------------
# A single "time alignment" script that works for BOTH NCEP and CMIP6 forcing,
# without hardcoding forcing grid resolution (H, W).
#
# This script keeps your *old pipeline logic* intact:
#   - Forcing timeline: 6-hourly, starts at year-10-25 00:00
#   - Surge (station CSV) timeline: hourly, starts at year-10-25 01:00
#   - Training centers: start at year-11-01 00:00, then 06:00, 12:00, 18:00, ...
#   - Max forcing history: 48 hours (8 steps of 6 hours) -> window length = 9 slices
#   - Label alignment:
#       forcing center t=00 -> surge hours [00..05]
#       forcing center t=06 -> surge hours [06..11]
#       forcing center t=12 -> surge hours [12..17]
#       forcing center t=18 -> surge hours [18..23]
#
# The only major upgrades (same as your current script):
#   (A) Infer forcing H/W from forcing shape: (T, H, W, 5)
#   (B) "peryear" cutoff is Option B:
#       - It does NOT require March to exist.
#       - It finds the last FULL day (00-23 coverage) available (common across stations),
#         capped by (year+1)-03-31 23:00.
#   (C) Output directories are ALWAYS model-specific:
#       Aligned_fixed315_hist48/<model_tag>/...
#       Aligned_peryear_hist48/<model_tag>/...
#
# -----------------------------------------------------------------------------
# NEW: Global-mean pressure feature (optional)
# -----------------------------------------------------------------------------
# We now assume forcing files are saved as NPZ:
#   forcing_local_YYYY.npz
#
# Required key:
#   - forcing: (T, H, W, 5)  where channel order is unchanged from before
#
# Optional key:
#   - p_mean_t: (T,)  spatial mean (over x,y) of ABSOLUTE pressure at each timestep
#
# IMPORTANT:
#   - We DO NOT change x/x_hist format (still 5 channels). train.py stays unchanged.
#   - If p_mean_t exists, we store it per-graph aligned with history window:
#       data.p_mean_hist: (WINDOW_LENGTH,)  (e.g. 9,)
#       data.p_mean_curr: scalar (the last element)
#   - If p_mean_t does not exist, the script behaves exactly like before.
#
# -----------------------------------------------------------------------------
# OUTPUTS
# -----------------------------------------------------------------------------
# For each (year, station), for both versions:
#   - fixed  : cutoff at (year+1)-03-15 (center up to 18:00, station up to 23:00)
#   - peryear: cutoff at computed last_full_day (center up to 18:00, station up to 23:00)
#
# Saved files:
#   <OUT_FIXED>/<model_tag>/dicts/<year>_<year+1>_<station>_fixed315_hist48_dict.npz
#   <OUT_FIXED>/<model_tag>/graphs/<year>_<year+1>_<station>_fixed315_hist48_graphs.pt
#
#   <OUT_PERYEAR>/<model_tag>/dicts/<year>_<year+1>_<station>_peryear_hist48_dict.npz
#   <OUT_PERYEAR>/<model_tag>/graphs/<year>_<year+1>_<station>_peryear_hist48_graphs.pt
#
# Each graphs.pt is a Python list of torch_geometric.data.Data objects.
#
# -----------------------------------------------------------------------------
# HOW TO RUN (examples)
# -----------------------------------------------------------------------------
#   python time_align_unified.py --model_tag NCEP
#   python time_align_unified.py --model_tag CMIP6_AWI
#   python time_align_unified.py --model_tag NCEP --years 1979:2014
#   python time_align_unified.py --model_tag NCEP --years 1979-1980 --debug_limit_stations 1
#
# If your folder structure differs, override:
#   python time_align_unified.py --model_tag NCEP \
#     --forcing_dir Forcing_Data/Processed_Forcing_NCEP \
#     --csv_dir     Surge_Data/NCEP_Reanalysis_fort63_station_CSVs
#
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


# =============================================================================
# Fixed constants (you said these do NOT change)
# =============================================================================
NUM_FEATURES = 5           # forcing feature channels (C) -- DO NOT CHANGE (train.py expects 5)
GT_PER_FORCING = 6         # 6 hourly station values per 6-hour forcing step
MAX_HISTORY_HOURS = 48     # fixed max history
assert MAX_HISTORY_HOURS % 6 == 0
MAX_HISTORY_STEPS = MAX_HISTORY_HOURS // 6    # 48h -> 8 forcing steps
WINDOW_LENGTH = MAX_HISTORY_STEPS + 1         # include current => 9 slices


# =============================================================================
# Model registry (convenience defaults)
# =============================================================================
MODEL_REGISTRY = {
    "NCEP": {
        "forcing_subdir": "Forcing_Data/Processed_Forcing_NCEP",
        "csv_subdir":     "Surge_Data/NCEP_Reanalysis_fort63_station_CSVs",
    },
    "CMIP6_AWI": {
        "forcing_subdir": "Forcing_Data/Processed_Forcing_CMIP6_AWI",
        "csv_subdir":     "Surge_Data/CMIP6_AWI_fort63_station_CSVs",
    },
    "CMIP6_CNRM": {
        "forcing_subdir": "Forcing_Data/Processed_Forcing_CMIP6_CNRM",
        "csv_subdir":     "Surge_Data/CMIP6_CNRM_fort63_station_CSVs",
    },
    "CMIP6_EC_EARTH": {
        "forcing_subdir": "Forcing_Data/Processed_Forcing_CMIP6_EC_EARTH",
        "csv_subdir":     "Surge_Data/CMIP6_EC_EARTH_fort63_station_CSVs",
    },
    "CMIP6_MPI": {
        "forcing_subdir": "Forcing_Data/Processed_Forcing_CMIP6_MPI",
        "csv_subdir":     "Surge_Data/CMIP6_MPI_fort63_station_CSVs",
    },
    "CMIP6_MRI": {
        "forcing_subdir": "Forcing_Data/Processed_Forcing_CMIP6_MRI",
        "csv_subdir":     "Surge_Data/CMIP6_MRI_fort63_station_CSVs",
    },
}


# =============================================================================
# Config container (for printing / clarity)
# =============================================================================
@dataclass
class AlignConfig:
    data_root: Path
    model_tag: str
    forcing_dir: Path
    csv_dir: Path
    out_root_fixed: Path
    out_root_peryear: Path
    stations: List[str]
    years: List[int]
    exclude_years: List[int]
    edge_mode: str
    max_edges: int
    debug_limit_years: Optional[int] = None
    debug_limit_stations: Optional[int] = None


# =============================================================================
# Time index builders (do NOT change your old assumptions)
# =============================================================================
def build_forcing_time_index(year: int, nsteps: int) -> np.ndarray:
    """
    Forcing time axis:
      - 6-hour interval
      - starts at year-10-25 00:00
    """
    t0 = datetime(year, 10, 25, 0, 0)
    return np.array([t0 + timedelta(hours=6 * i) for i in range(nsteps)])


def build_station_time_index(year: int, nsteps: int) -> np.ndarray:
    """
    Station time axis:
      - hourly interval
      - starts at year-10-25 01:00
      - row 0 corresponds to 10/25 01:00
    """
    t0 = datetime(year, 10, 25, 1, 0)
    return np.array([t0 + timedelta(hours=i) for i in range(nsteps)])


def validated_station_time_index(df: pd.DataFrame, year: int, csv_path: Path) -> np.ndarray:
    """Validate CSV times before using the historical fixed hourly timeline.

    Older CSVs without a ``time`` column retain the implicit-time convention.
    When timestamps are available, shifted origins, gaps, duplicates and invalid
    values must fail instead of silently assigning labels to different hours.
    """
    if df.empty:
        raise ValueError(f"Station CSV is empty: {csv_path}")

    expected = build_station_time_index(year, len(df))
    if "time" not in df.columns:
        print(
            f"[WARN] {csv_path} has no 'time' column; using the legacy assumption "
            f"of hourly rows starting at {expected[0]}."
        )
        return expected

    try:
        actual = pd.DatetimeIndex(pd.to_datetime(df["time"], errors="raise", utc=True)).tz_localize(None)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid station timestamps in {csv_path}: {exc}") from exc

    mismatches = np.flatnonzero(actual != pd.DatetimeIndex(expected))
    if len(mismatches):
        row = int(mismatches[0])
        raise ValueError(
            f"Station timestamp mismatch in {csv_path} at row {row}: "
            f"actual={actual[row]}, expected={expected[row]}. "
            "Expected continuous, unique hourly timestamps beginning at "
            f"{year}-10-25 01:00 UTC; check the origin, missing hours and duplicates."
        )
    return expected


# =============================================================================
# Forcing file discovery (NPZ only)
# =============================================================================
def discover_forcing_files(forcing_dir: Path) -> Dict[int, Path]:
    """
    Map {year -> forcing_file_path}.

    We ONLY use .npz forcing files now, because they can carry extra metadata
    (e.g., p_mean_t). This prevents accidental mismatch between forcing arrays
    and auxiliary per-timestep stats.

    Expected filename pattern:
      forcing_local_YYYY.npz
    """
    candidates = sorted(forcing_dir.glob("**/forcing_local_*.npz"))
    year_to_path: Dict[int, Path] = {}

    for p in candidates:
        m = re.search(r"forcing_local_(\d{4})\.npz$", p.name)
        if m:
            year_to_path[int(m.group(1))] = p

    return dict(sorted(year_to_path.items(), key=lambda kv: kv[0]))


def load_forcing_npz(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load forcing from a .npz.

    Required keys:
      - forcing: (T, H, W, 5)

    Optional keys:
      - p_mean_t: (T,)
        Meaning: spatial mean of ABSOLUTE pressure at each forcing timestep
        (computed during preprocessing over the SAME local region / grid).

    Returns:
      forcing: float32 array (T,H,W,5)
      p_mean_t: float32 array (T,) or None if missing
    """
    z = np.load(path, allow_pickle=False)
    if "forcing" not in z:
        raise KeyError(f"{path} missing key 'forcing'. Keys={list(z.keys())}")

    forcing = z["forcing"].astype(np.float32)
    p_mean_t = z["p_mean_t"].astype(np.float32) if "p_mean_t" in z else None
    return forcing, p_mean_t


def infer_grid_from_forcing(arr: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Forcing array must be shaped:
      (T, H, W, C)
    where C must equal NUM_FEATURES (=5).
    """
    if arr.ndim != 4:
        raise ValueError(f"Forcing must be 4D (T,H,W,C). Got shape={arr.shape}")
    T, H, W, C = arr.shape
    if C != NUM_FEATURES:
        raise ValueError(f"Expected forcing channels C={NUM_FEATURES}, got C={C}, shape={arr.shape}")
    return T, H, W, C


# =============================================================================
# Edge builders
# =============================================================================
def build_fully_connected_edge_index(num_nodes: int, include_self: bool = False) -> torch.Tensor:
    """
    Build a fully-connected directed edge_index.

    For N nodes:
      edges = N*(N-1) if include_self=False
      edges = N*N     if include_self=True
    """
    nodes = np.arange(num_nodes, dtype=np.int64)
    row = np.repeat(nodes, num_nodes)
    col = np.tile(nodes, num_nodes)

    if not include_self:
        mask = row != col
        row = row[mask]
        col = col[mask]

    edge_index = np.stack([row, col], axis=0)
    return torch.from_numpy(edge_index).long()


def build_grid4_edge_index(H: int, W: int) -> torch.Tensor:
    """
    Optional sparse edges: 4-neighborhood on the HxW grid.
    Node count H*W stays the same; only edges change.
    """
    def nid(r: int, c: int) -> int:
        return r * W + c

    edges: List[Tuple[int, int]] = []
    for r in range(H):
        for c in range(W):
            u = nid(r, c)
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    v = nid(rr, cc)
                    edges.append((u, v))

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_grid8_edge_index(H: int, W: int) -> torch.Tensor:
    """
    Optional sparse edges: 8-neighborhood on the HxW grid (grid4 + diagonals).
    Node count H*W stays the same; only edges change.
    """
    def nid(r: int, c: int) -> int:
        return r * W + c

    edges: List[Tuple[int, int]] = []
    nbrs = [
        (-1, 0), (1, 0), (0, -1), (0, 1),      # grid4
        (-1, -1), (-1, 1), (1, -1), (1, 1),    # diagonals
    ]

    for r in range(H):
        for c in range(W):
            u = nid(r, c)
            for dr, dc in nbrs:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    v = nid(rr, cc)
                    edges.append((u, v))

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def make_edge_index(edge_mode: str, H: int, W: int, max_edges: int) -> torch.Tensor:
    """
    Build edge_index AFTER H/W is known.

    edge_mode:
      - "fc"   : fully connected (default, matches old pipeline)
      - "grid4": sparse 4-neighborhood
      - "grid8": sparse 8-neighborhood (grid4 + diagonals)

    max_edges:
      Safety guard to prevent accidental huge fully-connected graphs.
    """
    num_nodes = H * W

    if edge_mode == "fc":
        num_edges = num_nodes * (num_nodes - 1)
        if num_edges > max_edges:
            raise ValueError(
                f"Fully-connected graph too large: nodes={num_nodes}, edges={num_edges:,} > max_edges={max_edges:,}\n"
                f"Use --edge_mode grid4/grid8 or increase --max_edges."
            )
        return build_fully_connected_edge_index(num_nodes, include_self=False)

    if edge_mode == "grid4":
        return build_grid4_edge_index(H, W)

    if edge_mode == "grid8":
        return build_grid8_edge_index(H, W)

    raise ValueError(f"Unknown edge_mode={edge_mode}. Use 'fc', 'grid4', or 'grid8'.")


# =============================================================================
# Option B: peryear cutoff (does NOT require March to exist)
# =============================================================================
def compute_last_full_day_for_year(year: int, stations: List[str], csv_dir: Path) -> datetime.date:
    """
    Compute a per-year cutoff date common across stations.

    Rules per station:
      - station implicit hourly timestamps start: year-10-25 01:00
      - cap usable timestamps to: season_cap = (year+1)-03-31 23:00
      - find the last timestamp <= season_cap
      - convert to last FULL day:
          if last_t.hour == 23 -> last full day = last_t.date()
          else                 -> last full day = (last_t - 1 day).date()
    Then return MIN across stations (common coverage).

    We also enforce that the last full day is not before Nov 1 of the season,
    because training centers start at Nov 1 00:00.
    """
    season_cap = datetime(year + 1, 3, 31, 23, 0)
    season_start_date = datetime(year, 11, 1, 0, 0).date()

    candidate_days: List[datetime.date] = []

    for station in stations:
        csv_path = csv_dir / f"{year}_{year+1}_{station}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV for year={year}, station={station}: {csv_path}")

        df = pd.read_csv(csv_path)
        t_station = validated_station_time_index(df, year, csv_path)

        valid_mask = np.array([t <= season_cap for t in t_station], dtype=bool)
        if not valid_mask.any():
            raise ValueError(
                f"No timestamps <= {season_cap} in {csv_path}. "
                f"First={t_station[0]}, last={t_station[-1]}"
            )

        last_t = t_station[np.where(valid_mask)[0][-1]]

        if last_t.hour < 23:
            candidate = (last_t - timedelta(days=1)).date()
        else:
            candidate = last_t.date()

        if candidate < season_start_date:
            raise ValueError(
                f"{csv_path} ends before season start. candidate={candidate}, season_start={season_start_date}"
            )

        if last_t.month < 3:
            print(f"[peryear cutoff] Year {year}, {station}: ends early at {last_t} -> last full day {candidate}")
        else:
            print(f"[peryear cutoff] Year {year}, {station}: last in-cap time {last_t} -> last full day {candidate}")

        candidate_days.append(candidate)

    last_day = min(candidate_days)
    print(f"[peryear cutoff] Year {year}: common last full day (capped by Mar 31) = {last_day}")
    return last_day


# =============================================================================
# Core builder: one (year, station, version)
# =============================================================================
def process_one_pair(
    *,
    year: int,
    station: str,
    forcing: np.ndarray,             # (T, H, W, 5)
    p_mean_t: Optional[np.ndarray],  # (T,) or None
    t_forcing: np.ndarray,           # (T,)
    edge_index: torch.Tensor,
    H: int,
    W: int,
    version: str,                    # "fixed" or "peryear"
    out_root_fixed: Path,
    out_root_peryear: Path,
    csv_dir: Path,
    last_full_day: Optional[datetime.date] = None,
) -> None:
    """
    Build dict + graphs for one year-station pair.

    -------------------------------------------------------------------------
    CENTER TIMES and HISTORY
    -------------------------------------------------------------------------
    - Centers start at year-11-01 00:00 (6-hourly)
    - 48h history means we need forcing back to:
        (year-11-01 00:00) - 48h = year-10-30 00:00
      So we MUST crop forcing to start at 10/30 00:00 (or later),
      and never use forcing earlier than that for any sample.

    We implement that via:
      context_start = t_start_center - 48h
      forcing_ctx = forcing[(t_forcing >= context_start) & (t_forcing <= t_end_center)]

    -------------------------------------------------------------------------
    NEW: p_mean_t handling
    -------------------------------------------------------------------------
    If p_mean_t is present (length T), we crop it using the SAME mask as forcing,
    producing p_mean_ctx aligned 1-to-1 with forcing_ctx timesteps.

    Then for each graph i:
      - forcing window = forcing_ctx[start_idx:end_idx] has shape (WINDOW_LENGTH, H, W, 5)
      - p_mean window  = p_mean_ctx[start_idx:end_idx] has shape (WINDOW_LENGTH,)
    We attach:
      data.p_mean_hist (WINDOW_LENGTH,)
      data.p_mean_curr (scalar)
    """
    num_nodes = H * W

    # -------------------------
    # 1) Define center start and context start
    # -------------------------
    t_start_center = datetime(year, 11, 1, 0, 0)
    context_start = t_start_center - timedelta(hours=6 * MAX_HISTORY_STEPS)  # => year-10-30 00:00

    # -------------------------
    # 2) Define end times based on version
    # -------------------------
    if version == "fixed":
        t_end_center  = datetime(year + 1, 3, 15, 18, 0)
        t_end_station = datetime(year + 1, 3, 15, 23, 0)
        out_root = out_root_fixed
        suffix = "fixed315_hist48"

    elif version == "peryear":
        if last_full_day is None:
            raise ValueError("last_full_day must be provided for peryear version.")
        t_end_center  = datetime(last_full_day.year, last_full_day.month, last_full_day.day, 18, 0)
        t_end_station = datetime(last_full_day.year, last_full_day.month, last_full_day.day, 23, 0)
        out_root = out_root_peryear
        suffix = "peryear_hist48"

    else:
        raise ValueError(f"Unknown version={version}")

    # -------------------------
    # 3) Crop forcing (and p_mean_t if present) to [context_start, t_end_center]
    # -------------------------
    mask_force = (t_forcing >= context_start) & (t_forcing <= t_end_center)
    forcing_ctx = forcing[mask_force]
    t_forcing_ctx = t_forcing[mask_force]

    if forcing_ctx.shape[0] == 0:
        raise ValueError(
            f"No forcing in range [{context_start}, {t_end_center}] for year={year}, station={station}, version={version}"
        )

    # Keep p_mean aligned to forcing_ctx using the SAME mask
    p_mean_ctx: Optional[np.ndarray] = None
    if p_mean_t is not None:
        if p_mean_t.shape[0] != forcing.shape[0]:
            raise ValueError(
                f"p_mean_t length mismatch for year={year}: len(p_mean_t)={p_mean_t.shape[0]} vs forcing T={forcing.shape[0]}"
            )
        p_mean_ctx = p_mean_t[mask_force]
        if p_mean_ctx.shape[0] != forcing_ctx.shape[0]:
            raise RuntimeError("Internal error: p_mean_ctx and forcing_ctx time lengths differ.")

    # Center indices (within forcing_ctx) from Nov 1 00:00 to t_end_center
    center_mask = (t_forcing_ctx >= t_start_center) & (t_forcing_ctx <= t_end_center)
    center_indices_full = np.where(center_mask)[0]
    if len(center_indices_full) == 0:
        raise ValueError(f"No forcing centers found in forcing_ctx for year={year}, station={station}, version={version}")

    # Need at least MAX_HISTORY_STEPS steps before the first center
    if center_indices_full[0] < MAX_HISTORY_STEPS:
        raise ValueError(
            f"Not enough forcing history after crop for year={year}, station={station}, version={version}.\n"
            f"earliest center index in forcing_ctx={center_indices_full[0]}, need >= {MAX_HISTORY_STEPS}."
        )

    # -------------------------
    # 4) Load station CSV and crop to start at Nov 1 00:00
    # -------------------------
    csv_path = csv_dir / f"{year}_{year+1}_{station}.csv"
    df = pd.read_csv(csv_path)

    if not {"nc", "nc_tide"}.issubset(df.columns):
        raise KeyError(f"CSV {csv_path} must contain columns 'nc' and 'nc_tide'. Found: {list(df.columns)}")

    t_station_full = validated_station_time_index(df, year, csv_path)
    t_start_station = datetime(year, 11, 1, 0, 0)

    mask_station = (t_station_full >= t_start_station) & (t_station_full <= t_end_station)
    idx = np.where(mask_station)[0]
    if len(idx) == 0:
        raise ValueError(
            f"No station data in [{t_start_station}, {t_end_station}] for {csv_path}. "
            f"Station last time = {t_station_full[-1]}"
        )

    first_included_time = t_station_full[idx[0]]
    if first_included_time != t_start_station:
        raise ValueError(
            f"Station does not include exact Nov 1 00:00 start for {csv_path}.\n"
            f"First included={first_included_time}, expected={t_start_station}."
        )

    df_aligned = df.iloc[idx].copy().reset_index(drop=True)
    nc = df_aligned["nc"].to_numpy()
    nc_tide = df_aligned["nc_tide"].to_numpy()

    # -------------------------
    # 5) Determine how many samples N we can build
    # -------------------------
    N_raw = len(center_indices_full)
    max_blocks_station = len(nc) // GT_PER_FORCING
    N = min(N_raw, max_blocks_station)

    if N == 0:
        raise ValueError(
            f"N=0 for year={year}, station={station}, version={version} "
            f"(N_raw={N_raw}, max_blocks_station={max_blocks_station})."
        )

    center_indices = center_indices_full[:N]
    forcing_centers = forcing_ctx[center_indices]  # (N, H, W, 5)

    nc_used = nc[: N * GT_PER_FORCING]
    nc_tide_used = nc_tide[: N * GT_PER_FORCING]

    p_mean_status = "present" if p_mean_ctx is not None else "missing"
    print(
        f"{version.upper():7s} {year} {station:8s} | "
        f"N={N:4d} | forcing_centers={tuple(forcing_centers.shape)} | "
        f"nodes={num_nodes:5d} | edges={edge_index.shape[1]:,} | station_hours={len(nc_used)} | "
        f"p_mean_t={p_mean_status} | end_center={t_end_center}"
    )

    # -------------------------
    # 6) Save dict npz (centers + used station arrays)
    #     NOTE: We intentionally do NOT change this dict format.
    # -------------------------
    dict_obj = {
        "Forcing": forcing_centers,
        "nc": nc_used,
        "nc_tide": nc_tide_used,
        "meta_max_history_hours": MAX_HISTORY_HOURS,
        "meta_grid_H": H,
        "meta_grid_W": W,
        "meta_num_features": NUM_FEATURES,
        "meta_gt_per_forcing": GT_PER_FORCING,
        "meta_version": version,
        "meta_center_start": str(t_start_center),
        "meta_center_end": str(t_end_center),
        "meta_station_start": str(t_start_station),
        "meta_station_end": str(t_end_station),
        "meta_context_start": str(context_start),
    }

    dict_name = f"{year}_{year+1}_{station}_{suffix}_dict.npz"
    dict_path = out_root / "dicts" / dict_name
    np.savez(dict_path, **dict_obj)

    # -------------------------
    # 7) Build graphs list
    # -------------------------
    graphs: List[Data] = []

    # Sanity: first forcing center should equal Nov 1 00:00
    t0_center = t_forcing_ctx[center_indices[0]]
    if t0_center != t_start_center:
        print(f"[Warning] First forcing center time = {t0_center}, expected = {t_start_center}")

    for i in range(N):
        center_idx_local = center_indices[i]
        center_time = t_forcing_ctx[center_idx_local]

        start_idx = center_idx_local - MAX_HISTORY_STEPS
        end_idx = center_idx_local + 1  # inclusive window => python slice end is exclusive

        window = forcing_ctx[start_idx:end_idx]  # (WINDOW_LENGTH, H, W, 5)
        if window.shape[0] != WINDOW_LENGTH:
            raise RuntimeError(
                f"forcing window length mismatch at i={i}: got {window.shape[0]}, expected {WINDOW_LENGTH}"
            )

        x_hist_np = window.reshape(WINDOW_LENGTH, num_nodes, NUM_FEATURES)
        x_hist = torch.from_numpy(x_hist_np).float()  # (9, num_nodes, 5)
        x_curr = x_hist[-1]                           # (num_nodes, 5)

        # Label block alignment
        s0 = i * GT_PER_FORCING
        s1 = (i + 1) * GT_PER_FORCING

        expected_center = t_start_station + timedelta(hours=s0)
        if center_time != expected_center:
            raise ValueError(
                f"Time alignment mismatch at i={i} for {year} {station} {version}:\n"
                f"  forcing center_time = {center_time}\n"
                f"  expected (station_start + {s0}h) = {expected_center}\n"
                f"Check forcing/station time origin assumptions."
            )

        nc_block = torch.from_numpy(nc_used[s0:s1]).float()               # (6,)
        nc_tide_block = torch.from_numpy(nc_tide_used[s0:s1]).float()     # (6,)
        y_block = nc_block - nc_tide_block                                # (6,)

        data = Data(
            x=x_curr,
            x_hist=x_hist,
            edge_index=edge_index,
            y=y_block,
            nc=nc_block,
            nc_tide=nc_tide_block,
            time_index=i,
            hour_start=s0,
            max_history_hours=MAX_HISTORY_HOURS,
            grid_H=H,
            grid_W=W,
            center_time=str(center_time),
        )

        # -------------------------
        # NEW: store p_mean history aligned to the SAME forcing window
        # (does not affect train.py unless you explicitly use it)
        # -------------------------
        if p_mean_ctx is not None:
            p_hist = p_mean_ctx[start_idx:end_idx]  # (WINDOW_LENGTH,)
            if p_hist.shape[0] != WINDOW_LENGTH:
                raise RuntimeError(
                    f"p_mean history length mismatch at i={i}: got {p_hist.shape[0]}, expected {WINDOW_LENGTH}"
                )
            p_mean_hist_t = torch.from_numpy(p_hist.astype(np.float32))  # (9,)
            data.p_mean_hist = p_mean_hist_t
            data.p_mean_curr = p_mean_hist_t[-1]  # scalar tensor

        graphs.append(data)

    graph_name = f"{year}_{year+1}_{station}_{suffix}_graphs.pt"
    graph_path = out_root / "graphs" / graph_name
    torch.save(graphs, graph_path)

    print("  Saved dict  :", dict_path)
    print("  Saved graphs:", graph_path, f"(num_graphs={len(graphs)})")


# =============================================================================
# CLI helpers
# =============================================================================
def parse_years_arg(s: str) -> List[int]:
    """
    Parse --years argument.

    Supported:
      - "1979:1999" => [1979..1998] (end exclusive)
      - "1979-1998" => [1979..1998] (inclusive)
      - "1979,1980,1981"
      - "1979"
    """
    s = s.strip()
    if ":" in s:
        a, b = s.split(":")
        return list(range(int(a), int(b)))
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    if "," in s:
        return [int(x) for x in s.split(",")]
    return [int(s)]


def resolve_dirs(
    data_root: Path,
    model_tag: str,
    forcing_dir: Optional[str],
    csv_dir: Optional[str],
) -> Tuple[Path, Path]:
    """
    Resolve forcing_dir and csv_dir.

    - If user passes overrides, use them.
    - Otherwise, use MODEL_REGISTRY defaults.

    Overrides can be:
      - relative paths (relative to data_root)
      - absolute paths
    """
    if forcing_dir is not None:
        fdir = (data_root / forcing_dir).resolve() if not Path(forcing_dir).is_absolute() else Path(forcing_dir).resolve()
    else:
        if model_tag not in MODEL_REGISTRY:
            raise KeyError(f"Unknown model_tag={model_tag}. Add to MODEL_REGISTRY or pass --forcing_dir.")
        fdir = (data_root / MODEL_REGISTRY[model_tag]["forcing_subdir"]).resolve()

    if csv_dir is not None:
        cdir = (data_root / csv_dir).resolve() if not Path(csv_dir).is_absolute() else Path(csv_dir).resolve()
    else:
        if model_tag not in MODEL_REGISTRY:
            raise KeyError(f"Unknown model_tag={model_tag}. Add to MODEL_REGISTRY or pass --csv_dir.")
        cdir = (data_root / MODEL_REGISTRY[model_tag]["csv_subdir"]).resolve()

    if not fdir.exists():
        raise FileNotFoundError(f"Forcing dir not found: {fdir}")
    if not cdir.exists():
        raise FileNotFoundError(f"CSV dir not found: {cdir}")

    return fdir, cdir


def make_output_roots(
    data_root: Path,
    model_tag: str,
    out_root_fixed: Optional[str],
    out_root_peryear: Optional[str],
) -> Tuple[Path, Path]:
    """
    Output folder policy: ALWAYS connect outputs with model_tag.

    Default bases:
      - Aligned_fixed315_hist48
      - Aligned_peryear_hist48

    Actual outputs:
      - Aligned_fixed315_hist48/<model_tag>/
      - Aligned_peryear_hist48/<model_tag>/

    If user passes --out_root_fixed / --out_root_peryear:
      - treat relative paths as relative to data_root
      - treat absolute paths as absolute
      - still append /<model_tag> to ensure model separation
    """
    if out_root_fixed is not None:
        base_fixed = (data_root / out_root_fixed).resolve() if not Path(out_root_fixed).is_absolute() else Path(out_root_fixed).resolve()
    else:
        base_fixed = (data_root / "Aligned_fixed315_hist48").resolve()

    if out_root_peryear is not None:
        base_peryear = (data_root / out_root_peryear).resolve() if not Path(out_root_peryear).is_absolute() else Path(out_root_peryear).resolve()
    else:
        base_peryear = (data_root / "Aligned_peryear_hist48").resolve()

    r_fixed = (base_fixed / model_tag).resolve()
    r_peryear = (base_peryear / model_tag).resolve()

    for root in [r_fixed, r_peryear]:
        (root / "dicts").mkdir(parents=True, exist_ok=True)
        (root / "graphs").mkdir(parents=True, exist_ok=True)

    return r_fixed, r_peryear


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", type=str, default=".", help="Path to Lehigh_Data_Processed (default: cwd)")
    ap.add_argument("--model_tag", type=str, default="NCEP", help="e.g., NCEP, CMIP6_AWI, ...")

    ap.add_argument("--forcing_dir", type=str, default=None, help="Override forcing directory (relative to data_root or absolute)")
    ap.add_argument("--csv_dir", type=str, default=None, help="Override station CSV directory (relative to data_root or absolute)")

    ap.add_argument("--out_root_fixed", type=str, default=None, help="Override base output dir for fixed (model_tag will be appended)")
    ap.add_argument("--out_root_peryear", type=str, default=None, help="Override base output dir for peryear (model_tag will be appended)")

    ap.add_argument("--stations", type=str, nargs="+", default=["Lewes", "Boston", "CBBT", "Battery"])

    ap.add_argument(
        "--years",
        type=str,
        default=None,
        help="Examples: 1979:1999 (end exclusive), 1979-1998 (inclusive), 1979,1980,1981, or 1979. "
             "If omitted, infer years from forcing files.",
    )
    ap.add_argument("--exclude_years", type=int, nargs="*", default=[])

    ap.add_argument("--edge_mode", type=str, default="grid4", choices=["fc", "grid4", "grid8"])
    ap.add_argument("--max_edges", type=int, default=50_000_000)

    ap.add_argument("--debug_limit_years", type=int, default=None, help="Process only first K years (debug)")
    ap.add_argument("--debug_limit_stations", type=int, default=None, help="Process only first K stations (debug)")

    args = ap.parse_args()

    # Resolve paths
    data_root = Path(args.data_root).resolve()
    forcing_dir, csv_dir = resolve_dirs(data_root, args.model_tag, args.forcing_dir, args.csv_dir)
    out_root_fixed, out_root_peryear = make_output_roots(data_root, args.model_tag, args.out_root_fixed, args.out_root_peryear)

    # Discover forcing files (NPZ only)
    year_to_path = discover_forcing_files(forcing_dir)
    if not year_to_path:
        raise FileNotFoundError(
            f"No forcing .npz files found under {forcing_dir}\n"
            f"Expected files like: forcing_local_1979.npz containing key 'forcing'."
        )

    # Determine years to process
    if args.years is None:
        years = sorted(year_to_path.keys())
    else:
        years = parse_years_arg(args.years)

    # Exclude years
    exclude = set(args.exclude_years or [])
    years = [y for y in years if y not in exclude]

    # Apply debug limits
    if args.debug_limit_years is not None:
        years = years[: args.debug_limit_years]
    stations = list(args.stations)
    if args.debug_limit_stations is not None:
        stations = stations[: args.debug_limit_stations]

    cfg = AlignConfig(
        data_root=data_root,
        model_tag=args.model_tag,
        forcing_dir=forcing_dir,
        csv_dir=csv_dir,
        out_root_fixed=out_root_fixed,
        out_root_peryear=out_root_peryear,
        stations=stations,
        years=years,
        exclude_years=list(exclude),
        edge_mode=args.edge_mode,
        max_edges=args.max_edges,
        debug_limit_years=args.debug_limit_years,
        debug_limit_stations=args.debug_limit_stations,
    )

    # Print config
    print("\n================= CONFIG =================")
    print("data_root       :", cfg.data_root)
    print("model_tag       :", cfg.model_tag)
    print("forcing_dir     :", cfg.forcing_dir)
    print("csv_dir         :", cfg.csv_dir)
    print("out_root_fixed  :", cfg.out_root_fixed)
    print("out_root_peryear:", cfg.out_root_peryear)
    print("stations        :", cfg.stations)
    print("years           :", cfg.years)
    print("exclude_years   :", cfg.exclude_years)
    print("edge_mode       :", cfg.edge_mode)
    print("max_edges       :", f"{cfg.max_edges:,}")
    print("MAX_HISTORY_HRS :", MAX_HISTORY_HOURS, "=> window length", WINDOW_LENGTH)
    print("=========================================\n")

    # Compute peryear cutoff (Option B) for each year
    year_last_day: Dict[int, datetime.date] = {}
    for y in cfg.years:
        year_last_day[y] = compute_last_full_day_for_year(y, cfg.stations, cfg.csv_dir)

    # Process each year
    for year in cfg.years:
        if year not in year_to_path:
            print(f"[Warning] No forcing file mapped for year={year} under {cfg.forcing_dir}, skipping.")
            continue

        forcing_path = year_to_path[year]
        forcing, p_mean_t = load_forcing_npz(forcing_path)
        T_full, H, W, C = infer_grid_from_forcing(forcing)
        t_forcing = build_forcing_time_index(year, T_full)

        # Build edge_index ONCE per year, reused for all station/version
        edge_index = make_edge_index(cfg.edge_mode, H, W, cfg.max_edges)

        last_day = year_last_day[year]

        print(f"\n=== Year {year} | forcing={forcing_path.name} | shape={forcing.shape} | peryear_last_day={last_day} ===")
        print(f"edge_index shape: {tuple(edge_index.shape)}  (nodes={H*W}, edges={edge_index.shape[1]:,})")

        for station in cfg.stations:
            # Fixed cutoff build (optional)
            # process_one_pair(
            #     year=year,
            #     station=station,
            #     forcing=forcing,
            #     p_mean_t=p_mean_t,
            #     t_forcing=t_forcing,
            #     edge_index=edge_index,
            #     H=H,
            #     W=W,
            #     version="fixed",
            #     out_root_fixed=cfg.out_root_fixed,
            #     out_root_peryear=cfg.out_root_peryear,
            #     csv_dir=cfg.csv_dir,
            # )

            # Peryear cutoff build (Option B)
            process_one_pair(
                year=year,
                station=station,
                forcing=forcing,
                p_mean_t=p_mean_t,  # NEW (optional)
                t_forcing=t_forcing,
                edge_index=edge_index,
                H=H,
                W=W,
                version="peryear",
                last_full_day=last_day,
                out_root_fixed=cfg.out_root_fixed,
                out_root_peryear=cfg.out_root_peryear,
                csv_dir=cfg.csv_dir,
            )

    # Sanity check: load one output file
    if cfg.years and cfg.stations:
        y0 = cfg.years[0]
        s0 = cfg.stations[0]
        example = cfg.out_root_peryear / "graphs" / f"{y0}_{y0+1}_{s0}_peryear_hist48_graphs.pt"
        print("\nSanity check file:", example)
        if example.exists():
            graphs = torch.load(example, weights_only=False, map_location="cpu")
            g0 = graphs[0]
            print("Loaded graphs:", len(graphs))
            print("x shape      :", tuple(g0.x.shape))
            print("x_hist shape :", tuple(g0.x_hist.shape), f"(should be {WINDOW_LENGTH}, {g0.x.shape[0]}, {NUM_FEATURES})")
            print("y shape      :", tuple(g0.y.shape))
            print("edge_index   :", tuple(g0.edge_index.shape))
            print("grid_H,grid_W:", getattr(g0, "grid_H", None), getattr(g0, "grid_W", None))
            print("center_time  :", getattr(g0, "center_time", None))
            if hasattr(g0, "p_mean_hist"):
                print("p_mean_hist  :", tuple(g0.p_mean_hist.shape), "(should be (9,))")
                print("p_mean_curr  :", g0.p_mean_curr.item() if torch.is_tensor(g0.p_mean_curr) else g0.p_mean_curr)
            else:
                print("p_mean_hist  : <not present> (your forcing npz did not include p_mean_t)")
        else:
            print("Example graph not found yet (maybe you processed different year/station first).")


if __name__ == "__main__":
    main()

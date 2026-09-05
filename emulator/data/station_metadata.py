"""Station metadata parsing and encoding helpers."""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn


def _try_load_station_json(station_json_dir: str, station_key: str) -> dict | None:
    """Load station metadata JSON using common filename variants."""
    if station_key is None or station_key == "":
        return None

    candidates = [
        os.path.join(station_json_dir, f"{station_key}.json"),
        os.path.join(station_json_dir, f"{station_key.lower()}.json"),
        os.path.join(station_json_dir, f"{station_key.upper()}.json"),
    ]

    for path in candidates:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    return None


def station_features_from_json(station: dict) -> torch.Tensor:
    """Convert station JSON fields into a numeric feature vector.

    The feature set includes scaled latitude/longitude/elevation and sin/cos
    encodings for latitude/longitude to improve periodic representation.
    """
    if station is None:
        return torch.zeros(0, dtype=torch.float32)

    def _get_float(keys, default=None):
        for key in keys:
            if key in station:
                try:
                    return float(station[key])
                except Exception:
                    pass
        return default

    lat = _get_float(["lat", "latitude", "Latitude"], default=0.0)
    lon = _get_float(["lon", "longitude", "Longitude"], default=0.0)
    elev = _get_float(["elevation_m", "elevation", "elev_m"], default=0.0)

    lat_norm = lat / 90.0
    lon_norm = lon / 180.0
    elev_norm = elev / 10.0

    features = [lat_norm, lon_norm, elev_norm]

    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    features += [np.sin(lat_rad), np.cos(lat_rad), np.sin(lon_rad), np.cos(lon_rad)]

    return torch.tensor(features, dtype=torch.float32)


class StationMetaEncoder(nn.Module):
    """Small MLP encoder for station metadata features."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(16, out_dim)

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)


__all__ = [
    "_try_load_station_json",
    "station_features_from_json",
    "StationMetaEncoder",
]

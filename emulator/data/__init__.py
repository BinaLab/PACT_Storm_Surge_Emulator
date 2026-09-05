"""Data layer for graph loading, splitting, normalization, and statistics."""

from .graph_store import (
    ForcingGraphStore,
    ForcingGraphView,
    make_all_years_test_indices,
    make_year_split_indices,
    years_from_indices,
)
from .loaders import build_loader
from .normalization import _as_feature_vector, normalize_inputs_inplace, normalize_targets
from .station_metadata import StationMetaEncoder, _try_load_station_json, station_features_from_json
from .stats import (
    _graph_has_pmean,
    compute_pmean_stats_from_store_rank0,
    compute_train_loss_thresholds_from_store,
    compute_x_mag_stats_from_store_rank0,
    compute_x_robust_stats_from_store_rank0,
    compute_x_stats_distributed_from_store,
    compute_y_stats_distributed_from_store,
)

__all__ = [
    "ForcingGraphStore",
    "ForcingGraphView",
    "make_all_years_test_indices",
    "make_year_split_indices",
    "years_from_indices",
    "build_loader",
    "_as_feature_vector",
    "normalize_inputs_inplace",
    "normalize_targets",
    "StationMetaEncoder",
    "_try_load_station_json",
    "station_features_from_json",
    "_graph_has_pmean",
    "compute_pmean_stats_from_store_rank0",
    "compute_train_loss_thresholds_from_store",
    "compute_x_mag_stats_from_store_rank0",
    "compute_x_robust_stats_from_store_rank0",
    "compute_x_stats_distributed_from_store",
    "compute_y_stats_distributed_from_store",
]

"""Model architectures used by the storm-surge emulator project."""

from .architectures import (
    MLP,
    PACT,
    PACTCNN,
    SpatialMLP0h,
    SpatialOnlyGraphSAGEBatch,
    TemporalCNN12h,
    TemporalLSTM12h,
    SpatioTemporalGraphSAGEBatch,
    TransformerBlock,
)

__all__ = [
    "MLP",
    "SpatialMLP0h",
    "SpatialOnlyGraphSAGEBatch",
    "TemporalCNN12h",
    "TemporalLSTM12h",
    "SpatioTemporalGraphSAGEBatch",
    "TransformerBlock",
    "PACT",
    "PACTCNN",
]

"""Model architectures used by the storm-surge emulator project."""

from .architectures import (
    MLP,
    PerceiverLikeModel3,
    SpatialOnlyGraphSAGEBatch,
    SpatioTemporalGraphSAGEBatch,
    TransformerBlock,
)

__all__ = [
    "MLP",
    "SpatialOnlyGraphSAGEBatch",
    "SpatioTemporalGraphSAGEBatch",
    "TransformerBlock",
    "PerceiverLikeModel3",
]

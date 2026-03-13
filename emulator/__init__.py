"""Core package for the storm-surge emulator codebase.

The package is intentionally split into domain-focused subpackages:
- `common`: runtime and distributed helpers.
- `data`: graph storage, splits, normalization, and statistics.
- `models`: model architectures.
- `training`: loss functions and train/eval engines.
- `inference`: inference-time metrics and execution helpers.
"""

__all__ = [
    "common",
    "data",
    "models",
    "training",
    "inference",
]

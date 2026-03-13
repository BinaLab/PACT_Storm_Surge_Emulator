"""File IO helpers with reproducibility-safe semantics."""

from __future__ import annotations

import json
import os


def write_json_atomic(path: str, obj: dict, indent: int = 2) -> None:
    """Write a JSON file atomically to prevent partial/corrupted outputs.

    Strategy:
    1. Write to `<path>.tmp`.
    2. Flush and fsync.
    3. Replace the target file atomically with `os.replace`.
    """
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


__all__ = ["write_json_atomic"]

"""Inference-time sample grouping helpers."""

from __future__ import annotations


def parse_year_tag(tag: str) -> tuple[int | None, int | None]:
    """Extract `(year0, year1)` from tags like `1979_1980_Battery_...`."""
    try:
        parts = str(tag).split("_")
        if len(parts) < 2:
            return (None, None)
        year0 = int(parts[0])
        year1 = int(parts[1])
        return (year0, year1)
    except Exception:
        return (None, None)


def classify_past_future(year0: int | None) -> str:
    """Classify by project-specific year ranges."""
    if year0 is None:
        return "other"
    if 1979 <= year0 <= 2014:
        return "past"
    if 2070 <= year0 <= 2099:
        return "future"
    return "other"


__all__ = ["parse_year_tag", "classify_past_future"]

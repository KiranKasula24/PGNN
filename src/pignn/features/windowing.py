"""Strict latest-window selection for node time-series."""
from __future__ import annotations


def latest_window(readings: list[dict], window_size: int) -> list[dict]:
    if window_size < 1:
        raise ValueError("window_size must be at least one")
    if not readings:
        raise ValueError("node time-series cannot be empty")
    ordered = sorted(readings, key=lambda item: item["t"])
    return ordered[-window_size:]

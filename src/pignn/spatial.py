"""Small, dependency-free spatial helpers for node/grid joins and local physics."""
from __future__ import annotations
import numpy as np


def local_metres(latitude: np.ndarray, longitude: np.ndarray, origin_latitude: float, origin_longitude: float) -> np.ndarray:
    """Project WGS84 points to local east/north metres around a declared origin."""
    metres_per_degree = 111_320.0
    return np.column_stack(((longitude - origin_longitude) * metres_per_degree * np.cos(np.deg2rad(origin_latitude)), (latitude - origin_latitude) * metres_per_degree)).astype(np.float32)


def nearest_grid_id(latitude: float, longitude: float, grid_rows: list[dict]) -> dict | None:
    """Return the nearest grid feature with usable coordinates; no invented join."""
    candidates = [row for row in grid_rows if row.get("latitude") is not None and row.get("longitude") is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda row: (float(row["latitude"]) - latitude) ** 2 + (float(row["longitude"]) - longitude) ** 2)

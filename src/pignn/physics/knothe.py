"""Knothe time-function for progressive subsidence."""
from __future__ import annotations

import numpy as np


def progression(time_days: np.ndarray | float, rate_per_day: float, start_day: float = 0.0) -> np.ndarray:
    """Return the bounded fraction of final deformation reached at each time."""
    if rate_per_day <= 0:
        raise ValueError("rate_per_day must be positive")
    t = np.asarray(time_days, dtype=float)
    elapsed = np.maximum(t - start_day, 0.0)
    return 1.0 - np.exp(-rate_per_day * elapsed)


def evolve(final_subsidence_mm: np.ndarray, time_days: np.ndarray | float, rate_per_day: float, start_day: float = 0.0) -> np.ndarray:
    """Scale a final deformation field into a [time, ...field] trajectory."""
    return progression(time_days, rate_per_day, start_day)[..., None, None] * np.asarray(final_subsidence_mm, dtype=float)

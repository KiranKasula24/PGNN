"""One Phase 6 postprocess call with the fixed API time-to-threshold shape."""
from __future__ import annotations

import numpy as np
from pignn.service.schemas import TimeToThreshold
from .uncertainty import uncertainty_range


DEFAULT_STABLE_SLOPE_EPSILON_MM_PER_DAY_SQUARED = 1e-3


def postprocess(time_days: np.ndarray, deformation_rate_samples_mm_per_day: np.ndarray, stable_slope_epsilon: float = DEFAULT_STABLE_SLOPE_EPSILON_MM_PER_DAY_SQUARED) -> tuple[str, TimeToThreshold]:
    result = uncertainty_range(time_days, deformation_rate_samples_mm_per_day)
    mean_rate = np.asarray(deformation_rate_samples_mm_per_day, dtype=float).mean(axis=0)
    trend_slope = np.polyfit(np.asarray(time_days, dtype=float), mean_rate, 1)[0]
    if stable_slope_epsilon < 0:
        raise ValueError("stable_slope_epsilon must be non-negative")
    trend = "stable" if abs(trend_slope) < stable_slope_epsilon else "accelerating" if trend_slope > 0 else "decelerating"
    return trend, TimeToThreshold(**result)

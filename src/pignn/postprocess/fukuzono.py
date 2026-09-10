"""Guarded Fukuzono inverse-velocity extrapolation."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class InverseVelocityForecast:
    time_to_threshold_days: float | None
    slope: float | None
    intercept: float | None


def inverse_velocity_forecast(time_days: np.ndarray, deformation_rate_mm_per_day: np.ndarray, minimum_points: int = 4) -> InverseVelocityForecast:
    """Fit 1/v = a*t+b and return future time-to-zero only when physically valid.

    This produces an advisory estimate, not a failure prediction. Non-positive,
    non-finite, decelerating, or already-passed trends deliberately return None.
    """
    time = np.asarray(time_days, dtype=float)
    rate = np.asarray(deformation_rate_mm_per_day, dtype=float)
    valid = np.isfinite(time) & np.isfinite(rate) & (rate > 0)
    time, rate = time[valid], rate[valid]
    if len(time) < minimum_points or np.unique(time).size < minimum_points:
        return InverseVelocityForecast(None, None, None)
    slope, intercept = np.polyfit(time, 1.0 / rate, 1)
    if slope >= 0:
        return InverseVelocityForecast(None, float(slope), float(intercept))
    failure_day = -intercept / slope
    remaining = failure_day - time.max()
    if not np.isfinite(remaining) or remaining <= 0:
        return InverseVelocityForecast(None, float(slope), float(intercept))
    return InverseVelocityForecast(float(remaining), float(slope), float(intercept))

"""One Phase 6 postprocess call with the fixed API time-to-threshold shape."""
from __future__ import annotations

import numpy as np
from pignn.service.schemas import TimeToThreshold
from .uncertainty import uncertainty_range


def postprocess(time_days: np.ndarray, deformation_rate_samples_mm_per_day: np.ndarray) -> tuple[str, TimeToThreshold]:
    result = uncertainty_range(time_days, deformation_rate_samples_mm_per_day)
    mean_rate = np.asarray(deformation_rate_samples_mm_per_day, dtype=float).mean(axis=0)
    trend_slope = np.polyfit(np.asarray(time_days, dtype=float), mean_rate, 1)[0]
    trend = "accelerating" if trend_slope > 0 else "decelerating" if trend_slope < 0 else "stable"
    return trend, TimeToThreshold(**result)

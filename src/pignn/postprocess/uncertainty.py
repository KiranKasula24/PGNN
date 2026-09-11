"""Ensemble-style uncertainty interval over inverse-velocity outputs."""
from __future__ import annotations

import numpy as np
from .fukuzono import inverse_velocity_forecast


def uncertainty_range(time_days: np.ndarray, rate_samples_mm_per_day: np.ndarray, lower_quantile: float = 0.1, upper_quantile: float = 0.9) -> dict:
    """Return an interval; confidence is forecast-valid sample fraction, not probability calibration."""
    samples = np.asarray(rate_samples_mm_per_day, dtype=float)
    if samples.ndim != 2:
        raise ValueError("rate_samples_mm_per_day must be [sample, time]")
    forecasts = [inverse_velocity_forecast(time_days, sample).time_to_threshold_days for sample in samples]
    valid = np.asarray([value for value in forecasts if value is not None], dtype=float)
    confidence = len(valid) / len(samples)
    if not len(valid):
        return {"low_days": None, "high_days": None, "confidence": 0.0}
    return {"low_days": float(np.quantile(valid, lower_quantile)), "high_days": float(np.quantile(valid, upper_quantile)), "confidence": float(confidence)}

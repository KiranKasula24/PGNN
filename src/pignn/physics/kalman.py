"""Canonical scalar Kalman filter shared by synthetic-data and firmware ports.

The Q/R defaults are provisional until hardware stillness tests provide measured
R values. Keep this exact recurrence aligned with the ESP32 implementation.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class KalmanTuning:
    process_variance: float
    measurement_variance: float
    initial_variance: float = 1.0


def filter_1d(measurements: np.ndarray, tuning: KalmanTuning) -> np.ndarray:
    values = np.asarray(measurements, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("measurements must be a non-empty 1-D series")
    if tuning.process_variance < 0 or tuning.measurement_variance <= 0:
        raise ValueError("Kalman variances must be non-negative Q and positive R")
    output = np.empty_like(values)
    estimate, variance = values[0], tuning.initial_variance
    output[0] = estimate
    for index in range(1, len(values)):
        variance += tuning.process_variance
        gain = variance / (variance + tuning.measurement_variance)
        estimate += gain * (values[index] - estimate)
        variance *= 1 - gain
        output[index] = estimate
    return output

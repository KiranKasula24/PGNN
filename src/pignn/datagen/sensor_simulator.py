"""Registry-driven virtual Tier-1 sensor simulation."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import numpy as np
from pignn.physics.kalman import KalmanTuning, filter_1d

SensorExtractor = Callable[[np.ndarray, np.ndarray, float, np.random.Generator], np.ndarray]
SENSOR_REGISTRY: dict[str, SensorExtractor] = {}


def register_sensor(name: str) -> Callable[[SensorExtractor], SensorExtractor]:
    def decorator(func: SensorExtractor) -> SensorExtractor:
        if name in SENSOR_REGISTRY:
            raise ValueError(f"sensor already registered: {name}")
        SENSOR_REGISTRY[name] = func
        return func
    return decorator


@register_sensor("displacement_filt")
def displacement(field_mm: np.ndarray, _times: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    return field_mm


@register_sensor("tilt_x_filt")
def tilt_x(field_mm: np.ndarray, _times: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    # PROVISIONAL proxy: replace with a spatial deformation-gradient/real IMU
    # forward model when deployment geometry and sensor calibration are known.
    return np.gradient(field_mm) / 1000.0 * 180 / np.pi


@register_sensor("tilt_y_filt")
def tilt_y(field_mm: np.ndarray, _times: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    return -0.7 * np.gradient(field_mm) / 1000.0 * 180 / np.pi


@register_sensor("vibration_filt")
def vibration(field_mm: np.ndarray, times: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    rate = np.gradient(field_mm, times, axis=0)
    return np.abs(rate) * 0.03


@dataclass(frozen=True)
class SensorNoise:
    """PROVISIONAL pre-hardware standard deviations; replace from measured Kalman R."""
    displacement_mm_std: float = 0.7
    tilt_deg_std: float = 0.015
    vibration_std: float = 0.08


def provisional_fuzzy_risk_index(displacement_filt: np.ndarray, times_days: np.ndarray) -> np.ndarray:
    """Placeholder, not a real Fuzzy Risk Index until firmware breakpoints exist."""
    return np.clip(np.abs(np.gradient(displacement_filt, times_days)) / 10.0, 0, 1)


def simulate_node_readings(field_by_time_mm: np.ndarray, times_days: np.ndarray, noise: SensorNoise, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Create named Tier-1 sensor readings for one node, preserving extension points."""
    specifications = (("displacement_filt", noise.displacement_mm_std), ("tilt_x_filt", noise.tilt_deg_std), ("tilt_y_filt", noise.tilt_deg_std), ("vibration_filt", noise.vibration_std))
    values: dict[str, np.ndarray] = {}
    for filtered_name, standard_deviation in specifications:
        truth = SENSOR_REGISTRY[filtered_name](field_by_time_mm, times_days, standard_deviation, rng)
        raw = truth + rng.normal(0, standard_deviation, truth.shape)
        values[filtered_name.replace("_filt", "_raw")] = raw
        values[filtered_name] = filter_1d(raw, KalmanTuning(process_variance=standard_deviation ** 2 * 0.05, measurement_variance=standard_deviation ** 2))
    values["risk_score"] = provisional_fuzzy_risk_index(values["displacement_filt"], times_days)
    return values

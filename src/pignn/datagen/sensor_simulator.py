"""Registry-driven virtual Tier-1 sensor simulation."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import numpy as np

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
    return field_mm + rng.normal(0, noise_std, field_mm.shape)


@register_sensor("tilt_x_filt")
def tilt_x(field_mm: np.ndarray, _times: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    # Node-level samples have no neighbouring spatial pixel; use a small,
    # deterministic proxy derived from its deformation progression.
    return np.gradient(field_mm) / 1000.0 * 180 / np.pi + rng.normal(0, noise_std, field_mm.shape)


@register_sensor("tilt_y_filt")
def tilt_y(field_mm: np.ndarray, _times: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    return -0.7 * np.gradient(field_mm) / 1000.0 * 180 / np.pi + rng.normal(0, noise_std, field_mm.shape)


@register_sensor("vibration_filt")
def vibration(field_mm: np.ndarray, times: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    rate = np.gradient(field_mm, times, axis=0)
    return np.abs(rate) * 0.03 + rng.normal(0, noise_std, field_mm.shape)


@dataclass(frozen=True)
class SensorNoise:
    displacement_mm_std: float = 0.7
    tilt_deg_std: float = 0.015
    vibration_std: float = 0.08


def simulate_node_readings(field_by_time_mm: np.ndarray, times_days: np.ndarray, noise: SensorNoise, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Create named Tier-1 sensor readings for one node, preserving extension points."""
    values = {
        "displacement_filt": SENSOR_REGISTRY["displacement_filt"](field_by_time_mm, times_days, noise.displacement_mm_std, rng),
        "tilt_x_filt": SENSOR_REGISTRY["tilt_x_filt"](field_by_time_mm, times_days, noise.tilt_deg_std, rng),
        "tilt_y_filt": SENSOR_REGISTRY["tilt_y_filt"](field_by_time_mm, times_days, noise.tilt_deg_std, rng),
        "vibration_filt": SENSOR_REGISTRY["vibration_filt"](field_by_time_mm, times_days, noise.vibration_std, rng),
    }
    values["risk_score"] = np.clip(np.abs(np.gradient(values["displacement_filt"], times_days)) / 10.0, 0, 1)
    return values

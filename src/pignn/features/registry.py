"""Small feature registry: new feature sources do not alter graph/model code."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import numpy as np

Extractor = Callable[[dict], tuple[float, bool]]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    unit: str
    extractor: Extractor


class FeatureRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, FeatureSpec] = {}

    def register(self, name: str, unit: str) -> Callable[[Extractor], Extractor]:
        def decorator(extractor: Extractor) -> Extractor:
            if name in self._specs:
                raise ValueError(f"feature already registered: {name}")
            self._specs[name] = FeatureSpec(name, unit, extractor)
            return extractor
        return decorator

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def extract(self, time_series: dict[str, list[dict]], window_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Return values/masks shaped [nodes, window, feature]. Invalid values are zeroed."""
        from .windowing import latest_window
        node_ids = sorted(time_series, key=lambda key: int(key))
        values = np.zeros((len(node_ids), window_size, len(self._specs)), dtype=np.float32)
        mask = np.zeros_like(values, dtype=np.float32)
        for node_index, node_id in enumerate(node_ids):
            readings = latest_window(time_series[node_id], window_size)
            offset = window_size - len(readings)
            for time_index, reading in enumerate(readings, start=offset):
                for feature_index, spec in enumerate(self._specs.values()):
                    value, valid = spec.extractor(reading)
                    if valid:
                        values[node_index, time_index, feature_index] = value
                        mask[node_index, time_index, feature_index] = 1.0
        return values, mask


DEFAULT_REGISTRY = FeatureRegistry()


def _numeric_field(name: str) -> Extractor:
    def extract(reading: dict) -> tuple[float, bool]:
        value = reading.get(name)
        if value is None or not np.isfinite(value):
            return 0.0, False
        return float(value), True
    return extract


for _name, _unit in (("tilt_x_filt", "degrees"), ("tilt_y_filt", "degrees"), ("vibration_filt", "hardware-TBD"), ("displacement_filt", "mm"), ("fuzzy_risk_index", "index")):
    DEFAULT_REGISTRY.register(_name, _unit)(_numeric_field(_name))


DEFAULT_REGISTRY.register("crack_anomaly_score", "0_to_1")(_numeric_field("crack_anomaly_score"))


@DEFAULT_REGISTRY.register("insar_los_displacement_mm", "mm")
def _insar_los(reading: dict) -> tuple[float, bool]:
    """Consume only pre-processed InSAR fields and preserve unavailable values."""
    from .insar_features import InSARSample
    return InSARSample(reading.get("insar_los_displacement_mm"), reading.get("insar_coherence")).validated()


DEFAULT_REGISTRY.register("insar_coherence", "0_to_1")(_numeric_field("insar_coherence"))

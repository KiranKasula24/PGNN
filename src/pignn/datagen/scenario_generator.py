"""Randomized PIM/Knothe scenarios in the locked dataset shape."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from uuid import uuid4
import numpy as np
from pignn.physics.knothe import progression
from pignn.physics.probability_integral import PIMParameters, PanelGeometry, basin_subsidence_mm
from .sensor_simulator import SensorNoise, simulate_node_readings


@dataclass(frozen=True)
class GenerationConfig:
    node_count: int = 12
    time_steps: int = 90
    time_step_days: float = 1.0
    dangerous_fraction: float = 0.10
    threshold_mm: float = 100.0
    noise: SensorNoise = SensorNoise()


def generate_scenario(rng: np.random.Generator, config: GenerationConfig = GenerationConfig()) -> dict:
    dangerous = rng.random() < config.dangerous_fraction
    panel = PanelGeometry(0.0, 0.0, float(rng.uniform(100, 250)), float(rng.uniform(180, 450)), float(rng.uniform(120, 350)), float(rng.uniform(0, 180)), float(rng.uniform(3.0, 5.0)))
    # Safe cases use a genuinely low-extraction parameter region rather than
    # being generated at 50/50 and discarded afterwards.
    params = PIMParameters(float(rng.uniform(0.55, 0.8) if dangerous else rng.uniform(0.01, 0.025)), float(rng.uniform(1.4, 2.3)), float(rng.uniform(-10, 10)))
    time_rate = float(rng.uniform(0.035, 0.09))
    positions = rng.uniform([-300, -300], [300, 300], size=(config.node_count, 2))
    times = np.arange(config.time_steps, dtype=float) * config.time_step_days
    final = basin_subsidence_mm(positions[:, 0], positions[:, 1], panel, params)
    true = progression(times, time_rate)[:, None] * final[None, :]
    time_series: dict[str, list[dict]] = {}
    for index, (x, y) in enumerate(positions):
        readings = simulate_node_readings(true[:, index], times, config.noise, rng)
        time_series[str(index + 1)] = [{"t": float(t), **{key: float(value[j]) for key, value in readings.items()}} for j, t in enumerate(times)]
    max_mm = float(final.max())
    reached = np.flatnonzero(true.max(axis=1) >= config.threshold_mm)
    return {
        "scenario_id": str(uuid4()), "source": "synthetic",
        "node_positions": [{"node_id": i + 1, "x_m": float(p[0]), "y_m": float(p[1]), "position_type": "synthetic"} for i, p in enumerate(positions)],
        "panel_geometry": asdict(panel), "pim_parameters": asdict(params), "knothe_rate_per_day": time_rate,
        "time_series": time_series,
        "label": {"subsidence_occurred": bool(max_mm >= config.threshold_mm), "max_subsidence_mm": max_mm, "time_to_threshold_days": float(times[reached[0]]) if len(reached) else None},
    }

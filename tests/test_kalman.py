import numpy as np
from pignn.datagen.sensor_simulator import SensorNoise, simulate_node_readings
from pignn.physics.kalman import KalmanTuning, filter_1d


def test_kalman_reduces_stationary_measurement_scatter():
    rng = np.random.default_rng(1)
    raw = 10 + rng.normal(0, 1, 200)
    filtered = filter_1d(raw, KalmanTuning(0.01, 1.0))
    assert filtered[30:].std() < raw[30:].std()


def test_sensor_simulation_keeps_raw_and_kalman_filtered_fields():
    readings = simulate_node_readings(np.linspace(0, 20, 20), np.arange(20.0), SensorNoise(), np.random.default_rng(2))
    assert {"displacement_raw", "displacement_filt", "tilt_x_raw", "tilt_x_filt"} <= readings.keys()
    assert not np.array_equal(readings["displacement_raw"], readings["displacement_filt"])

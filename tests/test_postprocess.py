import numpy as np
from pignn.postprocess.fukuzono import inverse_velocity_forecast
from pignn.postprocess.pipeline import postprocess
from pignn.postprocess.model_adapter import postprocess_pignn
from pignn.datagen.scenario_generator import GenerationConfig, generate_scenario
from pignn.graph.build_graph import scenario_to_pyg
from pignn.model.pignn import PIGNN


def test_fukuzono_known_linear_inverse_velocity_case():
    time = np.arange(0., 9.)
    rate = 1 / (10 - time)
    forecast = inverse_velocity_forecast(time, rate)
    assert forecast.time_to_threshold_days is not None
    assert abs(forecast.time_to_threshold_days - 2) < 1e-6


def test_postprocess_returns_fixed_time_threshold_shape():
    time = np.arange(0., 9.)
    samples = np.stack([1 / (10.0 + offset - time) for offset in (-0.25, 0, 0.25)])
    trend, threshold = postprocess(time, samples)
    assert trend == "accelerating"
    assert threshold.low_days is not None and threshold.high_days is not None
    assert threshold.low_days <= threshold.high_days
    assert threshold.confidence == 1.0


def test_nearly_flat_rates_are_classified_stable_with_tolerance():
    time = np.arange(10.0)
    rates = np.stack([np.full_like(time, 2.0) + 1e-6 * time, np.full_like(time, 2.0) - 1e-6 * time])
    trend, _ = postprocess(time, rates)
    assert trend == "stable"


def test_model_adapter_returns_api_time_threshold_shape():
    scenario = generate_scenario(np.random.default_rng(4), GenerationConfig(node_count=3, time_steps=8))
    graph = scenario_to_pyg(scenario, window_size=8)
    trend, threshold = postprocess_pignn(PIGNN(graph.x.shape[-1]), graph, np.arange(8.0), samples=3)
    assert trend in {"stable", "accelerating", "decelerating"}
    assert 0 <= threshold.confidence <= 1

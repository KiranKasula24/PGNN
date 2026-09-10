import numpy as np
from pignn.datagen.scenario_generator import GenerationConfig, generate_scenario
from pignn.graph.build_graph import scenario_to_pyg
from pignn.model.baseline import overfit_tiny_batch, train_validation_split


def test_baseline_overfits_a_tiny_balanced_batch():
    safe = GenerationConfig(node_count=12, time_steps=40, dangerous_fraction=0.0)
    danger = GenerationConfig(node_count=12, time_steps=40, dangerous_fraction=1.0)
    rng = np.random.default_rng(123)
    graphs = [scenario_to_pyg(generate_scenario(rng, safe), window_size=30) for _ in range(5)]
    graphs += [scenario_to_pyg(generate_scenario(rng, danger), window_size=30) for _ in range(5)]
    assert {int(graph.y.item()) for graph in graphs} == {0, 1}
    _, result = overfit_tiny_batch(graphs)
    assert result.final_loss < 0.03, result


def test_baseline_training_reduces_loss_on_synthetic_split():
    rng = np.random.default_rng(12)
    configs = [GenerationConfig(node_count=10, time_steps=35, dangerous_fraction=0.0), GenerationConfig(node_count=10, time_steps=35, dangerous_fraction=1.0)]
    graphs = [scenario_to_pyg(generate_scenario(rng, configs[index % 2]), 30) for index in range(20)]
    _, metrics = train_validation_split(graphs, epochs=80)
    assert metrics["train_loss"] < 0.3, metrics

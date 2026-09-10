import numpy as np
from pignn.datagen.scenario_generator import GenerationConfig, generate_scenario
from pignn.graph.build_graph import scenario_to_pyg
from pignn.model.pignn import overfit_tiny_batch


def test_pignn_overfits_a_tiny_balanced_batch():
    rng = np.random.default_rng(91)
    safe = GenerationConfig(node_count=6, time_steps=20, dangerous_fraction=0.0)
    danger = GenerationConfig(node_count=6, time_steps=20, dangerous_fraction=1.0)
    graphs = [scenario_to_pyg(generate_scenario(rng, safe), 15) for _ in range(3)]
    graphs += [scenario_to_pyg(generate_scenario(rng, danger), 15) for _ in range(3)]
    assert {int(graph.y.item()) for graph in graphs} == {0, 1}
    model, result = overfit_tiny_batch(graphs, epochs=250)
    assert result.final_loss < 0.06, result
    assert model.predicted_rate_mm_per_day(graphs[0]).shape == (6, 15)

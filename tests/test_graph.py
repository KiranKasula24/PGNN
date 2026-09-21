import numpy as np
import torch
from pignn.datagen.scenario_generator import GenerationConfig, generate_scenario
from pignn.graph.build_graph import physics_edges, scenario_to_pyg


def test_physics_edges_are_symmetric_and_fall_with_distance():
    edges, weights = physics_edges(np.array([[0., 0.], [10., 0.], [100., 0.]]), 100., 2.)
    pairs = {tuple(pair): weight.item() for pair, weight in zip(edges.T.tolist(), weights)}
    assert pairs[(0, 1)] == pairs[(1, 0)]
    assert pairs[(0, 1)] > pairs[(0, 2)] > 0


def test_scenario_becomes_pyg_data_with_masked_insar():
    scenario = generate_scenario(np.random.default_rng(2), GenerationConfig(node_count=4, time_steps=10))
    data = scenario_to_pyg(scenario, window_size=12)
    assert data.x.shape == (4, 12, 8)
    assert data.feature_mask.shape == data.x.shape
    assert data.edge_index.shape == (2, 12)
    assert data.edge_attr.shape == (12, 1)
    assert data.node_y.shape == (4,)
    assert torch.all((data.node_y >= 0) & (data.node_y <= 1))
    assert data.feature_mask[:, :, 6].sum() == 0  # no InSAR is intentionally no-data, never zero

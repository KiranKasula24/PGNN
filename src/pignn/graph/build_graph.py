"""Convert a generated scenario to PyG Data using fixed PIM influence edges."""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from pignn.features.registry import DEFAULT_REGISTRY, FeatureRegistry


def physics_edges(positions_m: np.ndarray, depth_m: float, tan_major_influence_angle: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Fully connected directed edges weighted by Gaussian PIM influence."""
    if len(positions_m) < 2:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1), dtype=torch.float32)
    sigma = depth_m / tan_major_influence_angle
    source, target, weights = [], [], []
    for i in range(len(positions_m)):
        for j in range(len(positions_m)):
            if i != j:
                distance = float(np.linalg.norm(positions_m[i] - positions_m[j]))
                source.append(i); target.append(j)
                weights.append(np.exp(-0.5 * (distance / sigma) ** 2))
    return torch.tensor([source, target], dtype=torch.long), torch.tensor(weights, dtype=torch.float32).unsqueeze(1)


def scenario_to_pyg(scenario: dict, window_size: int = 30, registry: FeatureRegistry = DEFAULT_REGISTRY) -> Data:
    values, mask = registry.extract(scenario["time_series"], window_size)
    nodes = sorted(scenario["node_positions"], key=lambda node: node["node_id"])
    positions = np.array([[node["x_m"], node["y_m"]] for node in nodes], dtype=np.float32)
    geometry, params = scenario["panel_geometry"], scenario["pim_parameters"]
    edge_index, edge_attr = physics_edges(positions, geometry["depth_m"], params["tan_major_influence_angle"])
    label = scenario.get("label") or {}
    node_severity = label.get("node_severity_0_to_1", {})
    return Data(
        x=torch.from_numpy(values), feature_mask=torch.from_numpy(mask), pos=torch.from_numpy(positions),
        edge_index=edge_index, edge_attr=edge_attr,
        y=torch.tensor([float(label.get("subsidence_occurred", False))], dtype=torch.float32),
        node_y=torch.tensor([float(node_severity.get(str(node["node_id"]), 0.0)) for node in nodes], dtype=torch.float32),
        node_ids=torch.tensor([node["node_id"] for node in nodes], dtype=torch.long),
        feature_names=list(registry.names), scenario_id=scenario["scenario_id"],
    )

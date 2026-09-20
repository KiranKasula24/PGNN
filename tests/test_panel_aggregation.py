from pignn.expected_state.config import load_mine_geometry, longwall_geometry_from_config
from pignn.panel_aggregation import aggregate_longwall_panel


def test_panel_aggregation_keeps_worst_associated_node_and_excludes_distant_nodes():
    geometry = longwall_geometry_from_config(load_mine_geometry())
    positions = {1: (0.0, 0.0), 2: (230.0, 0.0), 3: (2_000.0, 0.0)}
    result = aggregate_longwall_panel(geometry, positions, {1: 0.2, 2: 0.85, 3: 1.0})
    assert result.node_ids == [1, 2]
    assert result.severity_0_to_1 == 0.85

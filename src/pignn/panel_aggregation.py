"""Panel-level post-processing using the same Longwall angle-of-draw geometry."""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin, tan

from pignn.expected_state.longwall import LongwallGeometry


@dataclass(frozen=True)
class PanelAggregationResult:
    panel_id: str
    node_ids: list[int]
    severity_0_to_1: float


def _distance_to_panel_boundary_m(geometry: LongwallGeometry, x_m: float, y_m: float) -> float:
    """Distance from a point to the rotated panel rectangle; zero means inside."""
    angle = radians(geometry.orientation_deg)
    dx, dy = x_m - geometry.center_x_m, y_m - geometry.center_y_m
    local_x = dx * cos(angle) + dy * sin(angle)
    local_y = -dx * sin(angle) + dy * cos(angle)
    outside_x = max(abs(local_x) - geometry.width_m / 2, 0.0)
    outside_y = max(abs(local_y) - geometry.length_m / 2, 0.0)
    return (outside_x**2 + outside_y**2) ** 0.5


def associated_node_ids(geometry: LongwallGeometry, node_positions_m: dict[int, tuple[float, float]]) -> list[int]:
    """Associate nodes inside the panel's PIM major-influence zone."""
    influence_radius_m = geometry.depth_m / tan(radians(geometry.major_influence_angle_beta_deg))
    return [node_id for node_id, (x_m, y_m) in node_positions_m.items() if _distance_to_panel_boundary_m(geometry, x_m, y_m) <= influence_radius_m]


def aggregate_longwall_panel(geometry: LongwallGeometry, node_positions_m: dict[int, tuple[float, float]], node_severities: dict[int, float]) -> PanelAggregationResult:
    """Return a conservative max roll-up; it cannot be lower than its worst node."""
    node_ids = associated_node_ids(geometry, node_positions_m)
    invalid = {node_id: severity for node_id, severity in node_severities.items() if not 0 <= severity <= 1}
    if invalid:
        raise ValueError("node severities must be in [0, 1]")
    included_severities = [node_severities[node_id] for node_id in node_ids if node_id in node_severities]
    return PanelAggregationResult(geometry.panel_id, node_ids, max(included_severities, default=0.0))

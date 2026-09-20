"""Longwall PIM/Knothe expected-state calculation, independent of ML and sensors."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from math import tan, radians

import numpy as np

from pignn.physics.knothe import progression
from pignn.physics.probability_integral import PIMParameters, PanelGeometry, basin_subsidence_mm


@dataclass(frozen=True)
class LongwallGeometry:
    panel_id: str
    center_x_m: float
    center_y_m: float
    width_m: float
    length_m: float
    depth_m: float
    extraction_thickness_m: float
    subsidence_factor_a: float
    major_influence_angle_beta_deg: float
    knothe_time_coefficient_c_per_day: float
    extraction_start: datetime
    status: str = "active"
    current_face_position_m: float | None = None
    uncertainty_band_mm: float = 100.0
    orientation_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.status not in {"completed", "active", "planned"}:
            raise ValueError("status must be completed, active, or planned")
        if min(self.width_m, self.length_m, self.depth_m, self.extraction_thickness_m, self.uncertainty_band_mm) <= 0:
            raise ValueError("geometry dimensions and uncertainty band must be positive")
        if not 0 < self.subsidence_factor_a <= 1 or not 0 < self.major_influence_angle_beta_deg < 90:
            raise ValueError("invalid PIM parameters")
        if self.status == "active" and self.current_face_position_m is None:
            raise ValueError("active panels require current_face_position_m")


@dataclass(frozen=True)
class LongwallNodeState:
    node_id: int
    expected_deformation_mm: float
    observed_cumulative_displacement_mm: float | None
    divergence_mm: float | None
    physics_deviation_index: float | None


@dataclass(frozen=True)
class LongwallMonitoringResult:
    panel_id: str
    computed_at: datetime
    parameter_basis: dict[str, float | str]
    nodes: list[LongwallNodeState]
    is_hypothetical: bool = False


class LongwallExpectedStateEngine:
    """PIM plus Knothe expected field; observed values only enter after prediction.

    Multi-panel callers may sum separate expected fields. That is a documented
    linear-independence simplification, not a non-linear multi-seam model.
    """

    @staticmethod
    def expected_field(geometry: LongwallGeometry, node_positions_m: dict[int, tuple[float, float]], as_of: datetime) -> dict[int, float]:
        """Compute expected cumulative deformation without reading observations."""
        if as_of.tzinfo is None or geometry.extraction_start.tzinfo is None:
            raise ValueError("extraction_start and as_of must be timezone-aware")
        if geometry.status == "planned" or as_of < geometry.extraction_start:
            return {node_id: 0.0 for node_id in node_positions_m}
        face_position = geometry.length_m if geometry.status == "completed" else min(max(float(geometry.current_face_position_m or 0), 0.0), geometry.length_m)
        if face_position == 0:
            return {node_id: 0.0 for node_id in node_positions_m}
        elapsed_days = max((as_of - geometry.extraction_start).total_seconds() / 86_400.0, 0.0)
        face_speed_m_per_day = face_position / elapsed_days if elapsed_days else 0.0
        partial_panel = PanelGeometry(
            geometry.center_x_m,
            geometry.center_y_m - geometry.length_m / 2 + face_position / 2,
            geometry.width_m,
            face_position,
            geometry.depth_m,
            geometry.orientation_deg,
            geometry.extraction_thickness_m,
        )
        pim = PIMParameters(geometry.subsidence_factor_a, tan(radians(geometry.major_influence_angle_beta_deg)))
        result: dict[int, float] = {}
        for node_id, (x_m, y_m) in node_positions_m.items():
            local_y = y_m - (geometry.center_y_m - geometry.length_m / 2)
            reached_at_days = local_y / face_speed_m_per_day if face_speed_m_per_day > 0 else elapsed_days
            if local_y < 0 or local_y > face_position or reached_at_days > elapsed_days:
                result[node_id] = 0.0
                continue
            final_mm = float(basin_subsidence_mm(x_m, y_m, partial_panel, pim))
            result[node_id] = final_mm * float(progression(elapsed_days, geometry.knothe_time_coefficient_c_per_day, reached_at_days))
        return result

    @staticmethod
    def compare_observed(expected_mm: dict[int, float], observed_cumulative_mm: dict[int, float], uncertainty_band_mm: float) -> list[LongwallNodeState]:
        """Compare like-for-like cumulative values after physics calculation."""
        if uncertainty_band_mm <= 0:
            raise ValueError("uncertainty_band_mm must be positive")
        states = []
        for node_id, expected in expected_mm.items():
            observed = observed_cumulative_mm.get(node_id)
            divergence = None if observed is None else abs(observed - expected)
            states.append(LongwallNodeState(node_id, expected, observed, divergence, None if divergence is None else min(1.0, divergence / uncertainty_band_mm)))
        return states

    def monitoring_mode(self, geometry: LongwallGeometry, node_positions_m: dict[int, tuple[float, float]], observed_cumulative_mm: dict[int, float], as_of: datetime) -> LongwallMonitoringResult:
        expected = self.expected_field(geometry, node_positions_m, as_of)
        return LongwallMonitoringResult(
            geometry.panel_id,
            as_of,
            {"a": geometry.subsidence_factor_a, "beta_deg": geometry.major_influence_angle_beta_deg, "c_per_day": geometry.knothe_time_coefficient_c_per_day, "uncertainty_band_mm": geometry.uncertainty_band_mm, "source": "assumed_prototype"},
            self.compare_observed(expected, observed_cumulative_mm, geometry.uncertainty_band_mm),
        )

    def planning_mode(self, geometry: LongwallGeometry, node_positions_m: dict[int, tuple[float, float]], future_time_days: float) -> LongwallMonitoringResult:
        """Simulate a fully extracted hypothetical panel at a requested horizon."""
        if future_time_days < 0:
            raise ValueError("future_time_days must be non-negative")
        as_of = geometry.extraction_start + timedelta(days=future_time_days)
        simulated_geometry = replace(geometry, status="completed", current_face_position_m=None)
        expected = self.expected_field(simulated_geometry, node_positions_m, as_of)
        return LongwallMonitoringResult(
            geometry.panel_id, as_of,
            {"a": geometry.subsidence_factor_a, "beta_deg": geometry.major_influence_angle_beta_deg, "c_per_day": geometry.knothe_time_coefficient_c_per_day, "uncertainty_band_mm": geometry.uncertainty_band_mm, "source": "hypothetical_geometry"},
            self.compare_observed(expected, {}, geometry.uncertainty_band_mm),
            is_hypothetical=True,
        )

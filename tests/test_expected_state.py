from datetime import datetime, timezone

import pytest

from pignn.expected_state.config import load_mine_geometry, longwall_geometry_from_config
from pignn.expected_state.bord_and_pillar import BordAndPillarExpectedStateEngine
from pignn.expected_state.longwall import LongwallExpectedStateEngine
from pignn.mine_types import get_pipeline


def test_configured_mine_types_resolve_to_a_pipeline():
    longwall_engine, longwall_model = get_pipeline("longwall")
    bord_engine, bord_model = get_pipeline("bord_and_pillar")
    assert isinstance(longwall_engine, LongwallExpectedStateEngine)
    assert isinstance(bord_engine, BordAndPillarExpectedStateEngine)
    assert longwall_model.mine_type == "longwall"
    assert bord_model.mine_type == "bord_and_pillar"


def test_longwall_expected_equals_observed_has_zero_physics_deviation():
    config = load_mine_geometry()
    assert config["is_assumed"] is True
    geometry = longwall_geometry_from_config(config)
    engine = LongwallExpectedStateEngine()
    as_of = datetime(2026, 1, 1, tzinfo=timezone.utc)
    positions = {1: (0.0, -100.0), 2: (20.0, -50.0), 3: (0.0, 100.0)}
    expected = engine.expected_field(geometry, positions, as_of)
    result = engine.monitoring_mode(geometry, positions, expected, as_of)
    assert all(node.divergence_mm == pytest.approx(0.0) for node in result.nodes)
    assert all(node.physics_deviation_index == pytest.approx(0.0) for node in result.nodes)
    assert result.nodes[-1].expected_deformation_mm == 0.0  # Not reached by active face.


def test_longwall_planning_mode_is_explicitly_non_live():
    geometry = longwall_geometry_from_config(load_mine_geometry())
    result = LongwallExpectedStateEngine().planning_mode(
        geometry, {1: (0.0, -100.0)}, future_time_days=365
    )
    assert result.is_hypothetical is True
    assert result.nodes[0].observed_cumulative_displacement_mm is None

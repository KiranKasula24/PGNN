from datetime import datetime, timezone

from fastapi.testclient import TestClient

from pignn.expected_state.baseline import NodeBaseline, cumulative_displacement_since_baseline
from pignn.service.main import app


def test_cumulative_displacement_is_baseline_relative():
    baseline = NodeBaseline(7, baseline_displacement_mm=12.5, installed_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert cumulative_displacement_since_baseline(18.0, baseline) == 5.5


def test_expected_state_monitoring_and_whatif_are_separate_contracts():
    payload = {
        "nodes": [{"node_id": 1, "x_m": 0.0, "y_m": -100.0}],
        "observed_cumulative_displacement_mm": {"1": 0.0},
        "as_of": "2026-01-01T00:00:00Z",
    }
    client = TestClient(app, headers={"X-GEOARGUS-INTERNAL-KEY": "test-internal-key"})
    monitoring = client.post("/expected-state/state", json=payload)
    hypothetical = client.post("/expected-state/whatif", json={
        "nodes": payload["nodes"], "future_time_days": 365,
        "geometry_override": {"panel_id": "LW-ALTERNATE", "width_m": 350, "center_x_m": 100},
    })
    assert monitoring.status_code == 200
    assert monitoring.json()["is_hypothetical"] is False
    assert hypothetical.status_code == 200
    assert hypothetical.json()["is_hypothetical"] is True
    assert hypothetical.json()["panel_id"] == "LW-ALTERNATE"
    assert hypothetical.json()["nodes"][0]["observed_cumulative_displacement_mm"] is None
    context = client.get("/expected-state/context?latitude=20&longitude=80")
    assert context.status_code == 200
    assert context.json()["context"]["source"] == "simulated_prototype"


def test_bord_and_pillar_risk_and_panel_routes_are_http_reachable():
    client = TestClient(app, headers={"X-GEOARGUS-INTERNAL-KEY": "test-internal-key"})
    structural = client.post("/expected-state/bord-and-pillar/state")
    hypothetical = client.post("/expected-state/bord-and-pillar/whatif", json={"hypothetical_extraction_percent": 0.7})
    synthesis = client.post("/risk-synthesis", json={"risk_input_a": 0.8, "risk_input_b": 0.1})
    panel = client.post("/expected-state/panel-aggregation", json={
        "nodes": [{"node_id": 1, "x_m": 0, "y_m": 0}], "node_severities": {"1": 0.7},
    })
    assert structural.status_code == 200
    assert structural.json()["cphsr"]["coefficients_verified"] is False
    assert hypothetical.status_code == 200
    assert hypothetical.json()["is_hypothetical"] is True
    assert synthesis.status_code == 200
    assert synthesis.json()["confidence_badge"] == "disagreeing"
    assert panel.status_code == 200
    assert panel.json()["severity_0_to_1"] == 0.7

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
    client = TestClient(app)
    monitoring = client.post("/expected-state/state", json=payload)
    hypothetical = client.post("/expected-state/whatif", json={"nodes": payload["nodes"], "future_time_days": 365})
    assert monitoring.status_code == 200
    assert monitoring.json()["is_hypothetical"] is False
    assert hypothetical.status_code == 200
    assert hypothetical.json()["is_hypothetical"] is True
    assert hypothetical.json()["nodes"][0]["observed_cumulative_displacement_mm"] is None
    context = client.get("/expected-state/context?latitude=20&longitude=80")
    assert context.status_code == 200
    assert context.json()["context"]["source"] == "simulated_prototype"

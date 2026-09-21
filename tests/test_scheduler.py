import asyncio
from datetime import datetime, timezone
import json

import pytest

from pignn.expected_state.longwall import LongwallMonitoringResult, LongwallNodeState
from pignn.service.scheduler import ONE_MINUTE_SECONDS, PredictionScheduler, SupabaseClient, SupabaseSettings, build_longwall_monitoring_state, build_prediction_requests
from pignn.service.schemas import Prediction, TimeToThreshold, ZoneEntry

NODES = [{"node_id": 7, "site_id": "alpha", "mock_latitude": 20.0, "mock_longitude": 80.0}]
READINGS = [{"node_id": 7, "timestamp": "2026-09-11T10:00:00+00:00", "tilt_x_filt": 0.1, "tilt_y_filt": -0.1, "vibration_filt": 0.2, "displacement_filt": 1.2, "fuzzy_risk_index": 0.3, "crack_anomaly_score": 0.4}]
INSAR = [{"node_id": 7, "mock_latitude": 20.0, "mock_longitude": 80.0, "insar_los_velocity_mm": 2.5, "insar_coherence": 0.9, "raster_date": "2026-09-10"}]


class FakeSupabaseClient:
    def __init__(self) -> None:
        self.inserted = []

    def source_rows(self):
        return NODES, READINGS, INSAR

    def insert_prediction(self, site_id, prediction):
        self.inserted.append((site_id, prediction))


def test_build_requests_preserves_sensor_and_los_displacement_values():
    request = build_prediction_requests(NODES, READINGS, INSAR)[0]
    assert request.site_id == "alpha"
    assert request.nodes[0].readings[0].features["insar_los_displacement_mm"].value == 2.5
    assert request.nodes[0].readings[1].features["displacement_filt"].value == 1.2
    assert request.nodes[0].readings[1].features["crack_anomaly_score"].value == 0.4


def test_scheduler_runs_every_minute_and_writes_directly_to_predictions():
    client = FakeSupabaseClient()

    def predict(request):
        return Prediction(predicted_zone=[ZoneEntry(node_id=7, severity_0_to_1=0.4)], trend="stable", time_to_threshold=TimeToThreshold(confidence=0), model_version="test")

    scheduler = PredictionScheduler(client, predict)
    assert scheduler.interval_seconds == ONE_MINUTE_SECONDS
    asyncio.run(scheduler.run_once())
    assert client.inserted[0][0] == "alpha"
    assert scheduler.last_completed_at is not None
    assert scheduler.last_error is None


def test_scheduler_accepts_existing_application_supabase_environment_names(monkeypatch):
    monkeypatch.delenv("PIGNN_SUPABASE_URL", raising=False)
    monkeypatch.delenv("PIGNN_SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.setenv("NEXT_PUBLIC_SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-key")
    assert SupabaseSettings.from_env().url == "https://project.supabase.co"


def test_supabase_client_builds_authenticated_json_http_request(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return b'[{"node_id": 7}]'

    def fake_urlopen(request, timeout):
        captured["request"], captured["timeout"] = request, timeout
        return Response()

    monkeypatch.setattr("pignn.service.scheduler.urlopen", fake_urlopen)
    client = SupabaseClient(SupabaseSettings(url="https://project.supabase.co", service_role_key="secret"))
    assert client._request("POST", "predictions", {"score": 0.4}, prefer="return=minimal") == [{"node_id": 7}]
    request = captured["request"]
    assert request.full_url == "https://project.supabase.co/rest/v1/predictions"
    assert request.get_method() == "POST"
    assert json.loads(request.data) == {"score": 0.4}
    assert request.get_header("Apikey") == "secret"
    assert request.get_header("Authorization") == "Bearer secret"
    assert request.get_header("Prefer") == "return=minimal"
    assert captured["timeout"] == 15


def test_supabase_client_persists_monitoring_state_but_refuses_hypothetical_state(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def read(self): return b""

    def fake_urlopen(request, timeout):
        captured["request"] = request
        return Response()

    monkeypatch.setattr("pignn.service.scheduler.urlopen", fake_urlopen)
    client = SupabaseClient(SupabaseSettings(url="https://project.supabase.co", service_role_key="secret"))
    result = LongwallMonitoringResult(
        "LW-A", datetime(2026, 1, 1, tzinfo=timezone.utc),
        {"a": 0.7, "uncertainty_band_mm": 100.0, "source": "assumed_prototype"},
        [LongwallNodeState(7, 22.0, 20.0, 2.0, 0.02)],
    )
    client.insert_longwall_expected_state(result)
    assert captured["request"].full_url.endswith("/rest/v1/twin_state")
    assert json.loads(captured["request"].data)[0]["physics_deviation_index"] == 0.02
    with pytest.raises(ValueError):
        client.insert_longwall_expected_state(LongwallMonitoringResult(result.panel_id, result.computed_at, result.parameter_basis, result.nodes, is_hypothetical=True))


def test_monitoring_state_requires_registration_baseline_and_current_displacement():
    nodes = [{"node_id": 7, "mine_x_m": 0.0, "mine_y_m": -100.0}]
    readings = [{"node_id": 7, "recorded_at": "2026-01-01T00:00:00+00:00", "displacement_filt": 15.0}]
    baselines = [{"node_id": 7, "baseline_displacement_mm": 12.0}]
    result = build_longwall_monitoring_state(nodes, readings, baselines, datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert result is not None
    assert result.nodes[0].observed_cumulative_displacement_mm == 3.0
    assert build_longwall_monitoring_state([{"node_id": 7}], readings, baselines) is None

import asyncio

from pignn.service.scheduler import ONE_MINUTE_SECONDS, PredictionScheduler, SupabaseSettings, build_prediction_requests
from pignn.service.schemas import Prediction, TimeToThreshold, ZoneEntry

NODES = [{"node_id": 7, "site_id": "alpha", "mock_latitude": 20.0, "mock_longitude": 80.0}]
READINGS = [{"node_id": 7, "timestamp": "2026-09-11T10:00:00+00:00", "tilt_x_filt": 0.1, "tilt_y_filt": -0.1, "vibration_filt": 0.2, "displacement_filt": 1.2, "fuzzy_risk_index": 0.3}]
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


def test_scheduler_runs_every_minute_and_writes_directly_to_predictions():
    client = FakeSupabaseClient()

    def predict(request):
        return Prediction(predicted_zone=[ZoneEntry(node_id=7, severity_0_to_1=0.4)], trend="stable", time_to_threshold=TimeToThreshold(confidence=0), model_version="test")

    scheduler = PredictionScheduler(client, predict)
    assert scheduler.interval_seconds == ONE_MINUTE_SECONDS
    asyncio.run(scheduler.run_once())
    assert client.inserted[0][0] == "alpha"


def test_scheduler_accepts_existing_application_supabase_environment_names(monkeypatch):
    monkeypatch.delenv("PIGNN_SUPABASE_URL", raising=False)
    monkeypatch.delenv("PIGNN_SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.setenv("NEXT_PUBLIC_SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-key")
    assert SupabaseSettings.from_env().url == "https://project.supabase.co"

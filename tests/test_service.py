from datetime import datetime, timezone

import pytest
import torch
from fastapi.testclient import TestClient
from pignn.features.registry import DEFAULT_REGISTRY
from pignn.model.pignn import PIGNN
from pignn.service.inference import load_default_inference
from pignn.service.main import app


@pytest.fixture(autouse=True)
def isolated_checkpoint(tmp_path, monkeypatch):
    """Keep endpoint-contract tests independent of ignored deployment artifacts."""
    checkpoint = tmp_path / "pignn-test.pt"
    model = PIGNN(len(DEFAULT_REGISTRY.names))
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_count": len(DEFAULT_REGISTRY.names),
        "hidden_size": 32,
        "window_size": 8,
        "model_version": "pignn-0.1.1",
    }, checkpoint)
    monkeypatch.setenv("PIGNN_CHECKPOINT", str(checkpoint))
    load_default_inference.cache_clear()
    yield
    load_default_inference.cache_clear()


def test_predict_contract_is_stable():
    response = TestClient(app).post("/predict", json={"site_id": "demo", "nodes": [{"node_id": 1, "mock_latitude": 20.0, "mock_longitude": 80.0, "readings": [{"timestamp": datetime.now(timezone.utc).isoformat(), "features": {"displacement_filt": {"value": 1.2, "unit": "mm"}}}]}]})
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "pignn-0.1.1"
    assert body["trend"] in {"stable", "accelerating", "decelerating"}
    assert body["predicted_zone"][0]["node_id"] == 1
    assert 0 <= body["predicted_zone"][0]["severity_0_to_1"] <= 1
    assert set(body["time_to_threshold"]) == {"low_days", "high_days", "confidence"}


def test_predict_accepts_renamed_insar_feature():
    now = datetime.now(timezone.utc).isoformat()
    payload = {"site_id": "demo", "nodes": [{"node_id": 1, "mock_latitude": 20.0, "mock_longitude": 80.0, "readings": [{"timestamp": now, "features": {"insar_los_displacement_mm": {"value": 1.2, "unit": "mm"}, "insar_coherence": {"value": 0.9}}}]}]}
    assert TestClient(app).post("/predict", json=payload).status_code == 200

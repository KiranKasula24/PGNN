from datetime import datetime, timezone
from fastapi.testclient import TestClient
from pignn.service.main import app


def test_predict_contract_is_stable():
    response = TestClient(app).post("/predict", json={"site_id": "demo", "nodes": [{"node_id": 1, "mock_latitude": 20.0, "mock_longitude": 80.0, "readings": [{"timestamp": datetime.now(timezone.utc).isoformat(), "features": {"displacement_filt": {"value": 1.2, "unit": "mm"}}}]}]})
    assert response.status_code == 200
    assert response.json() == {"predicted_zone": [{"node_id": 1, "severity_0_to_1": 0.0}], "trend": "stable", "time_to_threshold": {"low_days": None, "high_days": None, "confidence": 0.0}, "model_version": "contract-stub-0.1.0"}

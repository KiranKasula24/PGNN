from fastapi import FastAPI
from .schemas import Prediction, PredictionRequest, TimeToThreshold, ZoneEntry

app = FastAPI(title="PIGNN ML Service", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_mode": "phase-0-contract-stub"}


@app.post("/predict", response_model=Prediction)
def predict(request: PredictionRequest) -> Prediction:
    """Phase-0 stable response; Phase 7 replaces only this implementation."""
    return Prediction(
        predicted_zone=[ZoneEntry(node_id=node.node_id, severity_0_to_1=0.0) for node in request.nodes],
        trend="stable",
        time_to_threshold=TimeToThreshold(low_days=None, high_days=None, confidence=0.0),
        model_version="contract-stub-0.1.0",
    )

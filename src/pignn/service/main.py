from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from fastapi import FastAPI, Query
from pignn.expected_state.config import load_mine_geometry, longwall_geometry_from_config
from pignn.mine_types import get_pipeline
from pignn.physics.geology_priors import explain_simulated_priors
from .inference import load_default_inference
from .scheduler import PredictionScheduler, SupabaseClient, SupabaseSettings
from .schemas import ExpectedStateRequest, ExpectedStateWhatIfRequest, Prediction, PredictionRequest


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = SupabaseSettings.from_env()
    scheduler = None
    if settings:
        inference = load_default_inference()
        scheduler = PredictionScheduler(SupabaseClient(settings), inference.predict)
        scheduler.start()
    app.state.scheduler = scheduler
    yield
    if scheduler:
        await scheduler.stop()


app = FastAPI(title="PIGNN ML Service", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    scheduler = getattr(app.state, "scheduler", None)
    scheduler_status = "disabled" if scheduler is None else "error" if scheduler.last_error else "running"
    return {
        "status": "degraded" if scheduler_status == "error" else "ok",
        "model_mode": "checkpoint-backed",
        "scheduler": {
            "status": scheduler_status,
            "interval_seconds": scheduler.interval_seconds if scheduler else None,
            "last_started_at": scheduler.last_started_at if scheduler else None,
            "last_completed_at": scheduler.last_completed_at if scheduler else None,
            "last_error": scheduler.last_error if scheduler else None,
        },
    }


@app.post("/predict", response_model=Prediction)
def predict(request: PredictionRequest) -> Prediction:
    """Run checkpoint-backed PIGNN inference without changing the response schema."""
    return load_default_inference().predict(request)


@app.post("/expected-state/state")
def expected_state(request: ExpectedStateRequest) -> dict:
    """Internal Monitoring Mode; this endpoint deliberately does not persist data."""
    config = load_mine_geometry()
    if config["mine_type"] != "longwall":
        raise ValueError("the active mine type does not expose a Longwall deformation field")
    engine, _ = get_pipeline(config["mine_type"])
    geometry = longwall_geometry_from_config(config)
    as_of = request.as_of or datetime.now(timezone.utc)
    positions = {node.node_id: (node.x_m, node.y_m) for node in request.nodes}
    return asdict(engine.monitoring_mode(geometry, positions, request.observed_cumulative_displacement_mm, as_of))


@app.post("/expected-state/whatif")
def expected_state_whatif(request: ExpectedStateWhatIfRequest) -> dict:
    """Internal Planning Mode; its hypothetical result cannot enter live tables."""
    config = load_mine_geometry()
    if config["mine_type"] != "longwall":
        raise ValueError("the active mine type does not expose a Longwall deformation field")
    engine, _ = get_pipeline(config["mine_type"])
    geometry = longwall_geometry_from_config(config)
    positions = {node.node_id: (node.x_m, node.y_m) for node in request.nodes}
    return asdict(engine.planning_mode(geometry, positions, request.future_time_days))


@app.get("/expected-state/context")
def expected_state_context(latitude: float = Query(20.0, ge=-90, le=90), longitude: float = Query(80.0, ge=-180, le=180)) -> dict:
    """Return the explicitly simulated geology basis until Earth Engine is live."""
    return explain_simulated_priors(latitude, longitude)

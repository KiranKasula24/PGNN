from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Query
from pignn.expected_state.config import bord_and_pillar_inputs_from_config, load_mine_geometry, longwall_geometry_from_config
from pignn.mine_types import get_pipeline
from pignn.panel_aggregation import aggregate_longwall_panel
from pignn.physics.geology_priors import explain_simulated_priors
from pignn.risk_synthesis import synthesize
from .inference import load_default_inference
from .scheduler import PredictionScheduler, SupabaseClient, SupabaseSettings
from .schemas import BordAndPillarWhatIfRequest, ExpectedStateRequest, ExpectedStateWhatIfRequest, PanelAggregationRequest, Prediction, PredictionRequest, RiskSynthesisRequest


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = SupabaseSettings.from_env()
    scheduler = None
    if settings:
        config = load_mine_geometry()
        if config["mine_type"] == "longwall":
            _, model = get_pipeline("longwall")
            scheduler = PredictionScheduler(SupabaseClient(settings), model.predict)
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
    config = load_mine_geometry()
    if config["mine_type"] != "longwall":
        raise HTTPException(status_code=409, detail="/predict is the Longwall PIGNN route; use the Bord-and-Pillar expected-state endpoints")
    _, model = get_pipeline(config["mine_type"])
    return model.predict(request)


@app.post("/expected-state/state")
def expected_state(request: ExpectedStateRequest) -> dict:
    """Internal Monitoring Mode; this endpoint deliberately does not persist data."""
    config = load_mine_geometry()
    if config["mine_type"] != "longwall":
        raise HTTPException(status_code=409, detail="the active mine type does not expose a Longwall deformation field")
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
        raise HTTPException(status_code=409, detail="the active mine type does not expose a Longwall deformation field")
    engine, _ = get_pipeline(config["mine_type"])
    geometry = longwall_geometry_from_config(config)
    if request.geometry_override:
        geometry = replace(geometry, **request.geometry_override.model_dump(exclude_none=True))
    positions = {node.node_id: (node.x_m, node.y_m) for node in request.nodes}
    return asdict(engine.planning_mode(geometry, positions, request.future_time_days))


@app.get("/expected-state/context")
def expected_state_context(latitude: float = Query(20.0, ge=-90, le=90), longitude: float = Query(80.0, ge=-180, le=180)) -> dict:
    """Return the explicitly simulated geology basis until Earth Engine is live."""
    return explain_simulated_priors(latitude, longitude)


@app.post("/expected-state/bord-and-pillar/state")
def bord_and_pillar_state() -> dict:
    """Internal structural Monitoring Mode, independent of live sensor inputs."""
    config = load_mine_geometry()
    params, pillars, depth, strength, unit_weight = bord_and_pillar_inputs_from_config(config)
    engine, _ = get_pipeline("bord_and_pillar")
    return asdict(engine.monitoring_mode(params, pillars, depth, strength, unit_weight))


@app.post("/expected-state/bord-and-pillar/whatif")
def bord_and_pillar_whatif(request: BordAndPillarWhatIfRequest) -> dict:
    config = load_mine_geometry()
    params, pillars, depth, strength, unit_weight = bord_and_pillar_inputs_from_config(config)
    engine, _ = get_pipeline("bord_and_pillar")
    return asdict(engine.planning_mode(params, pillars, depth, strength, unit_weight, request.hypothetical_extraction_percent))


@app.post("/risk-synthesis")
def risk_synthesis(request: RiskSynthesisRequest) -> dict:
    return asdict(synthesize(request.risk_input_a, request.risk_input_b, request.low_days, request.high_days))


@app.post("/expected-state/panel-aggregation")
def expected_state_panel_aggregation(request: PanelAggregationRequest) -> dict:
    config = load_mine_geometry()
    if config["mine_type"] != "longwall":
        raise HTTPException(status_code=409, detail="panel aggregation geometry is currently implemented for Longwall")
    geometry = longwall_geometry_from_config(config)
    positions = {node.node_id: (node.x_m, node.y_m) for node in request.nodes}
    return asdict(aggregate_longwall_panel(geometry, positions, request.node_severities))

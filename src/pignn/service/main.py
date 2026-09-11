from contextlib import asynccontextmanager
from fastapi import FastAPI
from .inference import load_default_inference
from .scheduler import PredictionScheduler, SupabaseClient, SupabaseSettings
from .schemas import Prediction, PredictionRequest


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
def health() -> dict[str, str]:
    return {"status": "ok", "model_mode": "checkpoint-backed"}


@app.post("/predict", response_model=Prediction)
def predict(request: PredictionRequest) -> Prediction:
    """Run checkpoint-backed PIGNN inference without changing the response schema."""
    return load_default_inference().predict(request)

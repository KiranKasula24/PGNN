"""Registry adapter for the existing checkpoint-backed Longwall PIGNN."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pignn.service.schemas import Prediction, PredictionRequest


class LongwallPIGNNModel:
    """Lazy adapter keeps expected-state calls independent of checkpoint presence."""
    mine_type = "longwall"

    def predict(self, request: "PredictionRequest") -> "Prediction":
        from pignn.service.inference import load_default_inference
        return load_default_inference().predict(request)

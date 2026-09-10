"""Bridge PIGNN deformation-rate outputs into Phase 6 uncertainty processing."""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data
from pignn.model.pignn import PIGNN
from .pipeline import postprocess


def postprocess_pignn(model: PIGNN, graph: Data, time_days: np.ndarray, samples: int = 20) -> tuple[str, object]:
    """MC-dropout ensemble of mean cluster deformation-rate trajectories."""
    if samples < 2:
        raise ValueError("at least two MC-dropout samples are required")
    was_training = model.training
    model.train()  # activates dropout while retaining the fixed physics graph
    with torch.no_grad():
        rates = [model.predicted_rate_mm_per_day(graph).mean(dim=0).cpu().numpy() for _ in range(samples)]
    model.train(was_training)
    return postprocess(time_days, np.asarray(rates))

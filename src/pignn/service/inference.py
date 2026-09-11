"""Checkpoint-backed conversion from the public API contract to PIGNN output."""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from pignn.features.registry import DEFAULT_REGISTRY
from pignn.graph.build_graph import physics_edges
from pignn.model.pignn import PIGNN
from pignn.postprocess.model_adapter import postprocess_pignn
from .schemas import Prediction, PredictionRequest, TimeToThreshold, ZoneEntry


DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[3] / "checkpoints" / "pignn-0.1.1.pt"
# The request contract deliberately has no mine-panel geometry. These defaults
# keep edge construction explicit until calibrated site geometry is supplied.
DEFAULT_DEPTH_M = 200.0
DEFAULT_TAN_MAJOR_INFLUENCE_ANGLE = 2.0


def _local_positions_metres(request: PredictionRequest) -> np.ndarray:
    """Project mock WGS84 positions around their centroid for PIM edge weights."""
    latitude = np.asarray([node.mock_latitude for node in request.nodes], dtype=np.float32)
    longitude = np.asarray([node.mock_longitude for node in request.nodes], dtype=np.float32)
    latitude_0 = float(latitude.mean())
    metres_per_degree = 111_320.0
    return np.column_stack((
        (longitude - longitude.mean()) * metres_per_degree * np.cos(np.deg2rad(latitude_0)),
        (latitude - latitude.mean()) * metres_per_degree,
    )).astype(np.float32)


def request_to_graph(request: PredictionRequest, window_size: int) -> tuple[Data, np.ndarray]:
    """Create a masked fixed-physics graph and elapsed-day time axis from an API request."""
    feature_names = DEFAULT_REGISTRY.names
    feature_index = {name: index for index, name in enumerate(feature_names)}
    values = np.zeros((len(request.nodes), window_size, len(feature_names)), dtype=np.float32)
    mask = np.zeros_like(values)
    latest_timestamps = []
    for node_index, node in enumerate(request.nodes):
        readings = sorted(node.readings, key=lambda reading: reading.timestamp)[-window_size:]
        offset = window_size - len(readings)
        for step, reading in enumerate(readings, start=offset):
            latest_timestamps.append(reading.timestamp.timestamp())
            for name, sample in reading.features.items():
                index = feature_index.get(name)
                if index is None or not sample.valid or sample.value is None or not np.isfinite(sample.value):
                    continue
                values[node_index, step, index] = sample.value
                mask[node_index, step, index] = 1.0
    positions = _local_positions_metres(request)
    edge_index, edge_attr = physics_edges(positions, DEFAULT_DEPTH_M, DEFAULT_TAN_MAJOR_INFLUENCE_ANGLE)
    # A common elapsed axis is required by the rate postprocessor.  Inputs from
    # different nodes can arrive asynchronously, so use a uniform window when
    # timestamps do not provide enough distinct observations.
    timestamps = np.unique(np.asarray(latest_timestamps, dtype=float))
    if len(timestamps) >= 2:
        time_days = (timestamps - timestamps[-1]) / 86_400.0
        time_days -= time_days.min()
        if len(time_days) != window_size:
            time_days = np.linspace(float(time_days.min()), float(time_days.max()), window_size)
    else:
        time_days = np.arange(window_size, dtype=float)
    return Data(
        x=torch.from_numpy(values), feature_mask=torch.from_numpy(mask),
        pos=torch.from_numpy(positions), edge_index=edge_index, edge_attr=edge_attr,
        node_ids=torch.tensor([node.node_id for node in request.nodes], dtype=torch.long),
        feature_names=list(feature_names),
    ), time_days


class PIGNNInference:
    def __init__(self, checkpoint_path: str | Path = DEFAULT_CHECKPOINT) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"PIGNN checkpoint not found: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        required = {"model_state_dict", "feature_count", "hidden_size", "window_size", "model_version"}
        missing = required - checkpoint.keys()
        if missing:
            raise ValueError(f"checkpoint is missing fields: {sorted(missing)}")
        if checkpoint["feature_count"] != len(DEFAULT_REGISTRY.names):
            raise ValueError("checkpoint feature count does not match the registered feature contract")
        self.window_size = int(checkpoint["window_size"])
        self.model_version = str(checkpoint["model_version"])
        self.model = PIGNN(int(checkpoint["feature_count"]), hidden_size=int(checkpoint["hidden_size"]))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, request: PredictionRequest) -> Prediction:
        graph, time_days = request_to_graph(request, self.window_size)
        with torch.no_grad():
            node_logits, _, _ = self.model(graph)
            severities = torch.sigmoid(node_logits).cpu().tolist()
        trend, threshold = postprocess_pignn(self.model, graph, time_days)
        return Prediction(
            predicted_zone=[ZoneEntry(node_id=node.node_id, severity_0_to_1=float(severity)) for node, severity in zip(request.nodes, severities)],
            trend=trend,
            time_to_threshold=TimeToThreshold.model_validate(threshold),
            model_version=self.model_version,
        )


@lru_cache(maxsize=1)
def load_default_inference() -> PIGNNInference:
    """Load the checkpoint once per service process, rather than per request."""
    return PIGNNInference(os.getenv("PIGNN_CHECKPOINT", str(DEFAULT_CHECKPOINT)))

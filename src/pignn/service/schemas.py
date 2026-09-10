"""Frozen, registry-friendly entry and exit API contracts."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field


class FeatureSample(BaseModel):
    """An extensible named feature value; invalid is distinct from zero."""
    value: float | None = None
    valid: bool = True
    unit: str | None = None


class NodeReading(BaseModel):
    timestamp: datetime
    features: dict[str, FeatureSample] = Field(default_factory=dict)


class NodeInput(BaseModel):
    node_id: int
    mock_latitude: float
    mock_longitude: float
    readings: list[NodeReading] = Field(min_length=1)


class PredictionRequest(BaseModel):
    """Entry contract: a time window, mock locations, and named features."""
    model_config = ConfigDict(extra="forbid")
    site_id: str
    nodes: list[NodeInput] = Field(min_length=1)
    requested_at: datetime | None = None


class ZoneEntry(BaseModel):
    node_id: int
    severity_0_to_1: Annotated[float, Field(ge=0, le=1)]


class TimeToThreshold(BaseModel):
    low_days: Annotated[float | None, Field(ge=0)] = None
    high_days: Annotated[float | None, Field(ge=0)] = None
    confidence: Annotated[float, Field(ge=0, le=1)]


class Prediction(BaseModel):
    predicted_zone: list[ZoneEntry]
    trend: Literal["stable", "accelerating", "decelerating"]
    time_to_threshold: TimeToThreshold
    model_version: str

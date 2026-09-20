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


class ExpectedStateNodeInput(BaseModel):
    node_id: int
    x_m: float
    y_m: float


class ExpectedStateRequest(BaseModel):
    """Internal backend contract; coordinates are local metres until GNSS CRS lands."""
    nodes: list[ExpectedStateNodeInput] = Field(min_length=1)
    observed_cumulative_displacement_mm: dict[int, float] = Field(default_factory=dict)
    as_of: datetime | None = None


class ExpectedStateWhatIfRequest(BaseModel):
    """Planning-only horizon from a declared extraction start; never live state."""
    nodes: list[ExpectedStateNodeInput] = Field(min_length=1)
    future_time_days: Annotated[float, Field(ge=0)]
    geometry_override: "LongwallGeometryOverride | None" = None


class LongwallGeometryOverride(BaseModel):
    """Optional alternative panel geometry for an isolated Planning Mode scenario."""
    panel_id: str | None = None
    center_x_m: float | None = None
    center_y_m: float | None = None
    width_m: Annotated[float | None, Field(gt=0)] = None
    length_m: Annotated[float | None, Field(gt=0)] = None
    depth_m: Annotated[float | None, Field(gt=0)] = None
    extraction_thickness_m: Annotated[float | None, Field(gt=0)] = None
    subsidence_factor_a: Annotated[float | None, Field(gt=0, le=1)] = None
    major_influence_angle_beta_deg: Annotated[float | None, Field(gt=0, lt=90)] = None
    knothe_time_coefficient_c_per_day: Annotated[float | None, Field(gt=0)] = None
    uncertainty_band_mm: Annotated[float | None, Field(gt=0)] = None
    orientation_deg: float | None = None


class BordAndPillarWhatIfRequest(BaseModel):
    hypothetical_extraction_percent: Annotated[float, Field(ge=0, le=1)]


class RiskSynthesisRequest(BaseModel):
    risk_input_a: Annotated[float, Field(ge=0, le=1)]
    risk_input_b: Annotated[float, Field(ge=0, le=1)]
    low_days: Annotated[float | None, Field(ge=0)] = None
    high_days: Annotated[float | None, Field(ge=0)] = None


class PanelAggregationRequest(BaseModel):
    nodes: list[ExpectedStateNodeInput] = Field(min_length=1)
    node_severities: dict[int, Annotated[float, Field(ge=0, le=1)]]

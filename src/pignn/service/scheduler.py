"""Minute-based sensor/InSAR inference worker that writes predictions to Supabase."""
from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from pignn.expected_state.longwall import LongwallMonitoringResult
from .schemas import FeatureSample, NodeInput, NodeReading, Prediction, PredictionRequest

logger = logging.getLogger(__name__)
ONE_MINUTE_SECONDS = 60


@dataclass(frozen=True)
class SupabaseSettings:
    """The scheduler reads source measurements and writes only `predictions`."""
    url: str
    service_role_key: str
    nodes_table: str = "nodes"
    readings_table: str = "readings"
    insar_table: str = "insar_node_features"
    predictions_table: str = "predictions"
    history_limit: int = 5_000

    @classmethod
    def from_env(cls) -> "SupabaseSettings | None":
        url = os.getenv("PIGNN_SUPABASE_URL") or os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
        key = os.getenv("PIGNN_SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not url and not key:
            return None
        if not url or not key:
            raise RuntimeError("a Supabase URL and SUPABASE_SERVICE_ROLE_KEY are required")
        return cls(
            url=url.rstrip("/"), service_role_key=key,
            nodes_table=os.getenv("PIGNN_SUPABASE_NODES_TABLE", "nodes"),
            readings_table=os.getenv("PIGNN_SUPABASE_READINGS_TABLE", "readings"),
            insar_table=os.getenv("PIGNN_SUPABASE_INSAR_TABLE", "insar_node_features"),
            predictions_table=os.getenv("PIGNN_SUPABASE_PREDICTIONS_TABLE", "predictions"),
            history_limit=int(os.getenv("PIGNN_SCHEDULER_HISTORY_LIMIT", "5000")),
        )


class SupabaseClient:
    def __init__(self, settings: SupabaseSettings) -> None:
        self.settings = settings

    def _request(self, method: str, path: str, payload: object | None = None, prefer: str | None = None) -> Any:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"apikey": self.settings.service_role_key, "Authorization": f"Bearer {self.settings.service_role_key}", "Content-Type": "application/json"}
        if prefer:
            headers["Prefer"] = prefer
        request = Request(f"{self.settings.url}/rest/v1/{path}", data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=15) as response:
                raw = response.read()
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Supabase {method} {path} failed ({error.code}): {detail}") from error
        return json.loads(raw) if raw else None

    def source_rows(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        limit = self.settings.history_limit
        nodes = quote(self.settings.nodes_table, safe="_")
        readings = quote(self.settings.readings_table, safe="_")
        insar = quote(self.settings.insar_table, safe="_")
        return (
            self._request("GET", f"{nodes}?select=*&order=node_id.asc") or [],
            self._request("GET", f"{readings}?select=*&order=recorded_at.desc&limit={limit}") or [],
            self._request("GET", f"{insar}?select=*&order=raster_date.desc&limit={limit}") or [],
        )

    def insert_prediction(self, site_id: str, prediction: Prediction) -> None:
        table = quote(self.settings.predictions_table, safe="_")
        threshold = prediction.time_to_threshold
        self._request("POST", table, {
            "site_id": site_id, "model_version": prediction.model_version,
            "predicted_zone": [entry.model_dump() for entry in prediction.predicted_zone],
            "trend": prediction.trend, "time_to_threshold_low_days": threshold.low_days,
            "time_to_threshold_high_days": threshold.high_days, "confidence": threshold.confidence,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }, prefer="return=minimal")

    def baseline_rows(self) -> list[dict[str, Any]]:
        """Read one-time GNSS-registration baselines once the proposed table exists."""
        return self._request("GET", "node_baseline?select=*") or []

    def insert_longwall_expected_state(self, result: LongwallMonitoringResult) -> None:
        """Persist Monitoring Mode only; Planning Mode is structurally refused."""
        if result.is_hypothetical:
            raise ValueError("hypothetical expected-state results must never be persisted")
        records = [{
            "node_id": node.node_id,
            "mine_type": "longwall",
            "expected_deformation_mm": node.expected_deformation_mm,
            "observed_cumulative_displacement_mm": node.observed_cumulative_displacement_mm,
            "physics_deviation_index": node.physics_deviation_index,
            "uncertainty_band_mm": result.parameter_basis.get("uncertainty_band_mm"),
            "parameter_basis": result.parameter_basis,
            "computed_at": result.computed_at.isoformat(),
        } for node in result.nodes]
        if records:
            self._request("POST", "twin_state", records, prefer="return=minimal")


def _value(row: dict[str, Any], flat_name: str, object_name: str, nested_name: str) -> float | None:
    """Read the existing flat DB field or its ingest-contract JSON representation."""
    value = row.get(flat_name)
    if value is None and isinstance(row.get(object_name), dict):
        value = row[object_name].get(nested_name)
    return value


def _sensor_features(row: dict[str, Any]) -> dict[str, FeatureSample]:
    fields = {
        "tilt_x_filt": _value(row, "tilt_x_filt", "tilt", "x_filt"),
        "tilt_y_filt": _value(row, "tilt_y_filt", "tilt", "y_filt"),
        "vibration_filt": _value(row, "vibration_filt", "vibration", "filt"),
        "displacement_filt": _value(row, "displacement_filt", "displacement", "filt"),
        # The application-owned source table still uses its pre-rename column.
        # This is a field-name adapter only; the Fuzzy Risk Index value is intact.
        "fuzzy_risk_index": row.get("fuzzy_risk_index", row.get("risk_score")),
    }
    return {name: FeatureSample(value=value, valid=value is not None) for name, value in fields.items()}


def build_prediction_requests(nodes: list[dict[str, Any]], readings: list[dict[str, Any]], insar_rows: list[dict[str, Any]]) -> list[PredictionRequest]:
    """Merge stored measurements by node, preserving LOS displacement unchanged.

    `insar_los_velocity_mm` is the established database column name; for this
    integration it means LOS displacement in mm for an SLC pair. It maps directly
    to the ML feature name, with no numeric conversion or vertical inference.
    """
    node_map = {int(node["node_id"]): node for node in nodes}
    node_readings: dict[int, list[NodeReading]] = defaultdict(list)
    for row in readings:
        node_id = int(row["node_id"])
        timestamp = row.get("recorded_at") or row.get("timestamp")
        if node_id in node_map and timestamp:
            node_readings[node_id].append(NodeReading(timestamp=timestamp, features=_sensor_features(row)))
    for row in insar_rows:
        node_id = int(row["node_id"])
        if node_id not in node_map or not row.get("raster_date"):
            continue
        timestamp = str(row["raster_date"])
        if len(timestamp) == 10:
            timestamp += "T00:00:00+00:00"
        displacement, coherence = row.get("insar_los_velocity_mm"), row.get("insar_coherence")
        node_readings[node_id].append(NodeReading(timestamp=timestamp, features={
            "insar_los_displacement_mm": FeatureSample(value=displacement, valid=displacement is not None, unit="mm"),
            "insar_coherence": FeatureSample(value=coherence, valid=coherence is not None),
        }))
    sites: dict[str, list[NodeInput]] = defaultdict(list)
    for node_id, history in node_readings.items():
        node = node_map[node_id]
        site_id, latitude, longitude = node.get("site_id"), node.get("mock_latitude"), node.get("mock_longitude")
        if not site_id:
            raise ValueError(f"nodes row {node_id} has no site_id")
        if latitude is None or longitude is None:
            latest = next((row for row in insar_rows if int(row["node_id"]) == node_id), None)
            latitude = None if latest is None else latest.get("mock_latitude")
            longitude = None if latest is None else latest.get("mock_longitude")
        if latitude is None or longitude is None:
            raise ValueError(f"node {node_id} has no mock coordinates")
        sites[str(site_id)].append(NodeInput(node_id=node_id, mock_latitude=latitude, mock_longitude=longitude, readings=sorted(history, key=lambda item: item.timestamp)))
    return [PredictionRequest(site_id=site_id, nodes=site_nodes) for site_id, site_nodes in sites.items()]


class PredictionScheduler:
    """Cancellable periodic pass; it has no queue, job table, or upstream trigger."""
    def __init__(self, client: SupabaseClient, predict: Callable[[PredictionRequest], Prediction], interval_seconds: int = ONE_MINUTE_SECONDS) -> None:
        self.client, self.predict, self.interval_seconds = client, predict, interval_seconds
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()
        self.last_started_at: datetime | None = None
        self.last_completed_at: datetime | None = None
        self.last_error: str | None = None

    async def run_once(self) -> None:
        self.last_started_at = datetime.now(timezone.utc)
        try:
            nodes, readings, insar_rows = await asyncio.to_thread(self.client.source_rows)
            for request in build_prediction_requests(nodes, readings, insar_rows):
                prediction = await asyncio.to_thread(self.predict, request)
                await asyncio.to_thread(self.client.insert_prediction, request.site_id, prediction)
        except Exception as error:
            self.last_error = str(error)
            raise
        else:
            self.last_completed_at = datetime.now(timezone.utc)
            self.last_error = None

    async def _run(self) -> None:
        while not self._stopping.is_set():
            try:
                await self.run_once()
            except Exception:
                logger.exception("Scheduled PIGNN inference pass failed")
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=self.interval_seconds)
            except TimeoutError:
                pass

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="pignn-supabase-scheduler")

    async def stop(self) -> None:
        self._stopping.set()
        if self._task:
            await self._task

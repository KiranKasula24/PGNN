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
import numpy as np
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from pignn.expected_state.config import load_mine_geometry, longwall_geometry_from_config
from pignn.expected_state.longwall import LongwallExpectedStateEngine
from pignn.expected_state.longwall import LongwallMonitoringResult
from pignn.panel_aggregation import aggregate_longwall_panel
from pignn.risk_synthesis import RiskSynthesisResult, synthesize
from pignn.spatial import local_metres, nearest_grid_id
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
    insar_table: str = "insar_grid_features"
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
            insar_table=os.getenv("PIGNN_SUPABASE_INSAR_TABLE", "insar_grid_features"),
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

    def insert_prediction(self, site_id: str, prediction: Prediction, risk: RiskSynthesisResult | None = None, panel_id: str | None = None) -> None:
        table = quote(self.settings.predictions_table, safe="_")
        threshold = prediction.time_to_threshold
        self._request("POST", table, {
            "site_id": site_id, "panel_id": panel_id, "model_version": prediction.model_version,
            "predicted_zone": [entry.model_dump() for entry in prediction.predicted_zone],
            "trend": prediction.trend, "time_to_threshold_low_days": threshold.low_days,
            "time_to_threshold_high_days": threshold.high_days, "confidence": threshold.confidence,
            "risk_score": None if risk is None else risk.risk_score,
            "risk_level": None if risk is None else risk_level(risk.risk_score),
            "confidence_badge": None if risk is None else risk.confidence_badge,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }, prefer="return=minimal")

    def insert_longwall_expected_state(self, site_id: str, result: LongwallMonitoringResult, grid_ids_by_node: dict[int, str] | None = None) -> None:
        """Persist Monitoring Mode only; Planning Mode is structurally refused."""
        if result.is_hypothetical:
            raise ValueError("hypothetical expected-state results must never be persisted")
        records = [{
            "site_id": site_id,
            "node_id": node.node_id,
            "grid_id": None if grid_ids_by_node is None else grid_ids_by_node.get(node.node_id),
            "mode": "longwall",
            "expected_deformation_mm": node.expected_deformation_mm,
            "observed_value_mm": node.observed_cumulative_displacement_mm,
            "physics_deviation_index": node.physics_deviation_index,
            "uncertainty_band_mm": result.parameter_basis.get("uncertainty_band_mm"),
            "parameter_basis": result.parameter_basis,
            "is_hypothetical": False,
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
        # Tier-2-derived score; absent remains masked until firmware supplies it.
        "crack_anomaly_score": row.get("crack_anomaly_score"),
    }
    return {name: FeatureSample(value=value, valid=value is not None) for name, value in fields.items()}


def build_prediction_requests(nodes: list[dict[str, Any]], readings: list[dict[str, Any]], insar_rows: list[dict[str, Any]]) -> list[PredictionRequest]:
    """Merge stored measurements by node, preserving LOS displacement unchanged.

    Grid features are joined to registered nodes by nearest latitude/longitude.
    LOS pair displacement maps to the ML feature unchanged; it is never treated
    as cumulative displacement or a vertical component.
    """
    node_map = {int(node["node_id"]): node for node in nodes}
    node_readings: dict[int, list[NodeReading]] = defaultdict(list)
    for row in readings:
        node_id = int(row["node_id"])
        timestamp = row.get("recorded_at") or row.get("timestamp")
        if node_id in node_map and timestamp:
            node_readings[node_id].append(NodeReading(timestamp=timestamp, features=_sensor_features(row)))
    for node_id, node in node_map.items():
        if node.get("latitude") is None or node.get("longitude") is None:
            continue
        row = nearest_grid_id(float(node["latitude"]), float(node["longitude"]), insar_rows)
        if row is None or not row.get("date_b"):
            continue
        timestamp = str(row["date_b"])
        if len(timestamp) == 10:
            timestamp += "T00:00:00+00:00"
        displacement, coherence = row.get("los_displacement_mm"), row.get("coherence")
        valid = displacement is not None and row.get("coherence_class") != "LOW"
        node_readings[node_id].append(NodeReading(timestamp=timestamp, features={
            "insar_los_displacement_mm": FeatureSample(value=displacement, valid=valid, unit="mm"),
            "insar_coherence": FeatureSample(value=coherence, valid=coherence is not None),
        }))
    sites: dict[str, list[NodeInput]] = defaultdict(list)
    for node_id, history in node_readings.items():
        node = node_map[node_id]
        site_id, latitude, longitude = node.get("site_id"), node.get("latitude"), node.get("longitude")
        if not site_id:
            raise ValueError(f"nodes row {node_id} has no site_id")
        if latitude is None or longitude is None:
            raise ValueError(f"node {node_id} has no registered latitude/longitude")
        sites[str(site_id)].append(NodeInput(node_id=node_id, mock_latitude=latitude, mock_longitude=longitude, readings=sorted(history, key=lambda item: item.timestamp)))
    return [PredictionRequest(site_id=site_id, nodes=site_nodes) for site_id, site_nodes in sites.items()]


def build_longwall_monitoring_state(nodes: list[dict[str, Any]], readings: list[dict[str, Any]], as_of: datetime | None = None) -> LongwallMonitoringResult | None:
    """Build a live expected-state result only for fully registered nodes.

    Pairwise InSAR LOS values are intentionally not used here: the comparison
    requires baseline-relative cumulative displacement in the same units.
    """
    config = load_mine_geometry()
    if config["mine_type"] != "longwall":
        return None
    baseline_by_node = {int(row["node_id"]): float(row["baseline_displacement_mm"]) for row in nodes if row.get("node_id") is not None and row.get("baseline_displacement_mm") is not None}
    positions: dict[int, tuple[float, float]] = {}
    observed: dict[int, float] = {}
    latest_by_node: dict[int, dict[str, Any]] = {}
    for row in readings:
        if row.get("node_id") is None:
            continue
        node_id = int(row["node_id"])
        timestamp = str(row.get("recorded_at") or row.get("timestamp") or "")
        if node_id not in latest_by_node or timestamp > str(latest_by_node[node_id].get("recorded_at") or latest_by_node[node_id].get("timestamp") or ""):
            latest_by_node[node_id] = row
    registered = [node for node in nodes if node.get("latitude") is not None and node.get("longitude") is not None]
    if not registered:
        return None
    origin_latitude = sum(float(node["latitude"]) for node in registered) / len(registered)
    origin_longitude = sum(float(node["longitude"]) for node in registered) / len(registered)
    for node in registered:
        node_id = int(node["node_id"])
        if node_id not in baseline_by_node:
            continue
        reading = latest_by_node.get(node_id)
        if reading is None:
            continue
        current_displacement = _value(reading, "displacement_filt", "displacement", "filt")
        if current_displacement is None:
            continue
        positions[node_id] = tuple(local_metres(np.asarray([float(node["latitude"])]), np.asarray([float(node["longitude"])]), origin_latitude, origin_longitude)[0])
        observed[node_id] = float(current_displacement) - baseline_by_node[node_id]
    if not positions:
        return None
    return LongwallExpectedStateEngine().monitoring_mode(longwall_geometry_from_config(config), positions, observed, as_of or datetime.now(timezone.utc))


def risk_level(score: float) -> str:
    return "normal" if score < 0.30 else "watch" if score < 0.60 else "warning" if score < 0.80 else "critical"


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
            prediction_runs: list[tuple[PredictionRequest, Prediction]] = []
            for request in build_prediction_requests(nodes, readings, insar_rows):
                prediction = await asyncio.to_thread(self.predict, request)
                prediction_runs.append((request, prediction))
            monitoring = await asyncio.to_thread(build_longwall_monitoring_state, nodes, readings)
            deviation_by_node = {} if monitoring is None else {
                node.node_id: node.physics_deviation_index for node in monitoring.nodes if node.physics_deviation_index is not None
            }
            for request, prediction in prediction_runs:
                severities = [entry.severity_0_to_1 for entry in prediction.predicted_zone]
                deviations = [deviation_by_node[node.node_id] for node in request.nodes if node.node_id in deviation_by_node]
                risk = None if not severities or not deviations else synthesize(max(severities), max(deviations), prediction.time_to_threshold.low_days, prediction.time_to_threshold.high_days)
                positions = {node.node_id: tuple(local_metres(np.asarray([node.mock_latitude]), np.asarray([node.mock_longitude]), np.mean([item.mock_latitude for item in request.nodes]), np.mean([item.mock_longitude for item in request.nodes]))[0]) for node in request.nodes}
                panel_id = aggregate_longwall_panel(longwall_geometry_from_config(load_mine_geometry()), positions, {entry.node_id: entry.severity_0_to_1 for entry in prediction.predicted_zone}).panel_id
                await asyncio.to_thread(self.client.insert_prediction, request.site_id, prediction, risk, panel_id)
            if monitoring is not None:
                node_by_id = {int(node["node_id"]): node for node in nodes}
                grid_ids_by_node = {
                    node_id: grid["grid_id"]
                    for node_id, node in node_by_id.items()
                    if node.get("latitude") is not None and node.get("longitude") is not None
                    for grid in [nearest_grid_id(float(node["latitude"]), float(node["longitude"]), insar_rows)]
                    if grid is not None and grid.get("grid_id") is not None
                }
                site_id = next((request.site_id for request, _ in prediction_runs), None)
                if site_id is None:
                    raise ValueError("cannot persist expected state without a prediction site_id")
                await asyncio.to_thread(self.client.insert_longwall_expected_state, site_id, monitoring, grid_ids_by_node)
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

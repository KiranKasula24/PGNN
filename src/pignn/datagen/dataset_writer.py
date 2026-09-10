"""Validated, versioned Parquet output for synthetic scenarios."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


REQUIRED_KEYS = {"scenario_id", "source", "node_positions", "panel_geometry", "time_series", "label"}


def validate_scenario(scenario: dict) -> None:
    missing = REQUIRED_KEYS - scenario.keys()
    if missing:
        raise ValueError(f"scenario misses locked fields: {sorted(missing)}")
    if scenario["source"] not in {"synthetic", "real"}:
        raise ValueError("source must be synthetic or real")
    if not scenario["node_positions"] or not scenario["time_series"]:
        raise ValueError("scenario must include nodes and time series")
    if scenario["source"] == "synthetic" and scenario["label"] is None:
        raise ValueError("synthetic scenarios require labels")


def dataset_config_hash(config: object) -> str:
    serialized = json.dumps(config, default=lambda value: getattr(value, "__dict__", str(value)), sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:12]


def write_parquet(scenarios: Iterable[dict], output_dir: Path, config: object) -> Path:
    """Write one reproducible batch. Nested sections stay JSON to preserve the contract."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    scenario_list = list(scenarios)
    for scenario in scenario_list:
        validate_scenario(scenario)
    output_dir.mkdir(parents=True, exist_ok=True)
    version = dataset_config_hash(config)
    rows = [{
        "scenario_id": s["scenario_id"], "source": s["source"],
        "node_positions_json": json.dumps(s["node_positions"]), "panel_geometry_json": json.dumps(s["panel_geometry"]),
        "pim_parameters_json": json.dumps(s.get("pim_parameters", {})), "time_series_json": json.dumps(s["time_series"]),
        "label_json": json.dumps(s["label"]), "dataset_version": version,
    } for s in scenario_list]
    path = output_dir / f"synthetic_{version}_{len(rows)}.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    return path


def load_parquet(path: Path) -> list[dict]:
    """Restore contract-shaped scenarios from a Phase 2 Parquet batch."""
    import pyarrow.parquet as pq
    rows = pq.read_table(path).to_pylist()
    return [{
        "scenario_id": row["scenario_id"], "source": row["source"],
        "node_positions": json.loads(row["node_positions_json"]),
        "panel_geometry": json.loads(row["panel_geometry_json"]),
        "pim_parameters": json.loads(row["pim_parameters_json"]),
        "time_series": json.loads(row["time_series_json"]), "label": json.loads(row["label_json"]),
    } for row in rows]

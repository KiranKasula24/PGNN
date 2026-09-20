"""Typed loading of the assumed-or-surveyed mine geometry configuration."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

from .longwall import LongwallGeometry
from .bord_and_pillar import CPHSRParameters, Pillar


DEFAULT_MINE_GEOMETRY = Path(__file__).resolve().parents[3] / "config" / "mine_geometry.json"


def load_mine_geometry(path: str | Path = DEFAULT_MINE_GEOMETRY) -> dict:
    with Path(path).open(encoding="utf-8") as source:
        geometry = json.load(source)
    if geometry.get("mine_type") not in {"longwall", "bord_and_pillar"}:
        raise ValueError("mine_geometry.json has an unsupported mine_type")
    if not isinstance(geometry.get("is_assumed"), bool):
        raise ValueError("mine_geometry.json must declare is_assumed")
    return geometry


def longwall_geometry_from_config(config: dict) -> LongwallGeometry:
    if config.get("mine_type") != "longwall":
        raise ValueError("active mine_type is not longwall")
    raw = config["longwall"]
    boundary = raw["panel_boundary"]
    return LongwallGeometry(
        panel_id=raw["panel_id"], center_x_m=float(raw["center_x_m"]), center_y_m=float(raw["center_y_m"]),
        width_m=float(boundary["width_m"]), length_m=float(boundary["length_m"]), depth_m=float(raw["depth_m"]),
        extraction_thickness_m=float(raw["extraction_thickness_m"]), subsidence_factor_a=float(raw["subsidence_factor_a"]),
        major_influence_angle_beta_deg=float(raw["major_influence_angle_beta_deg"]),
        knothe_time_coefficient_c_per_day=float(raw["knothe_time_coefficient_c_per_day"]),
        extraction_start=datetime.fromisoformat(raw["extraction_start_date"]), status=raw["status"],
        current_face_position_m=None if raw.get("current_face_position_m") is None else float(raw["current_face_position_m"]),
        uncertainty_band_mm=float(raw["uncertainty_band_mm"]), orientation_deg=float(raw.get("orientation_deg", 0.0)),
    )


def bord_and_pillar_inputs_from_config(config: dict) -> tuple[CPHSRParameters, list[Pillar], float, float, float]:
    raw = config["bord_and_pillar"]
    params = CPHSRParameters(
        rock_to_soil_ratio=float(raw["rock_to_soil_ratio"]), depth_m=float(raw["depth_m"]),
        extraction_height_m=float(raw["seam_thickness_m"]), brittleness_index=float(raw["brittleness_index"]),
        rock_density_kg_m3=float(raw["rock_density_kg_m3"]),
    )
    pillars = [Pillar(item["pillar_id"], float(item["width_m"]), float(item["gallery_width_m"]), tuple(item["adjacent_pillar_ids"])) for item in raw["pillars"]]
    return params, pillars, float(raw["depth_m"]), float(raw["intact_strength_mpa"]), float(raw["unit_weight_mpa_per_m"])

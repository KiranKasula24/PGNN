"""Analytic rectangular-panel Probability Integral Method (PIM) basin."""
from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
import numpy as np


@dataclass(frozen=True)
class PanelGeometry:
    center_x_m: float
    center_y_m: float
    width_m: float
    length_m: float
    depth_m: float
    orientation_deg: float = 0.0
    extraction_thickness_m: float = 3.0


@dataclass(frozen=True)
class PIMParameters:
    subsidence_factor: float = 0.65
    tan_major_influence_angle: float = 1.8
    inflection_offset_m: float = 0.0


def basin_subsidence_mm(x_m: np.ndarray | float, y_m: np.ndarray | float, panel: PanelGeometry, params: PIMParameters) -> np.ndarray:
    """Return positive downward final subsidence in mm for a rectangular panel.

    The rectangle is convolved with a normalized separable Gaussian influence
    kernel. Its scale is depth/tan(beta), so influence narrows as the major
    influence angle steepens. This analytic form is stable and testable.
    """
    if min(panel.width_m, panel.length_m, panel.depth_m, panel.extraction_thickness_m) <= 0:
        raise ValueError("panel dimensions and extraction thickness must be positive")
    if not 0 < params.subsidence_factor <= 1 or params.tan_major_influence_angle <= 0:
        raise ValueError("invalid PIM parameters")
    x, y = np.broadcast_arrays(np.asarray(x_m, dtype=float), np.asarray(y_m, dtype=float))
    angle = np.deg2rad(panel.orientation_deg)
    dx, dy = x - panel.center_x_m, y - panel.center_y_m
    local_x = dx * np.cos(angle) + dy * np.sin(angle)
    local_y = -dx * np.sin(angle) + dy * np.cos(angle)
    sigma = panel.depth_m / params.tan_major_influence_angle
    scale = sqrt(2.0) * sigma
    x0, x1 = -panel.width_m / 2 + params.inflection_offset_m, panel.width_m / 2 + params.inflection_offset_m
    y0, y1 = -panel.length_m / 2, panel.length_m / 2
    erf_vec = np.vectorize(erf)
    x_fraction = 0.5 * (erf_vec((local_x - x0) / scale) - erf_vec((local_x - x1) / scale))
    y_fraction = 0.5 * (erf_vec((local_y - y0) / scale) - erf_vec((local_y - y1) / scale))
    final_mm = 1000.0 * panel.extraction_thickness_m * params.subsidence_factor
    return final_mm * x_fraction * y_fraction

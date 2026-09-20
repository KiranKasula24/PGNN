"""Temporary deterministic geology context until Earth Engine is connected."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GeologyContext:
    latitude: float
    longitude: float
    soil_class: str
    bulk_density_g_cm3: float
    ndvi: float
    slope_deg: float
    source: str
    uncertainty_source: str


@dataclass(frozen=True)
class LongwallParameterPriors:
    subsidence_factor_a: float
    major_influence_angle_beta_deg: float
    knothe_time_coefficient_c_per_day: float
    uncertainty_band_mm: float
    source: str


def simulated_geology_context(latitude: float, longitude: float) -> GeologyContext:
    """Return documented prototype values; never present these as Earth Engine data."""
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("latitude/longitude are outside WGS84 bounds")
    return GeologyContext(latitude, longitude, "sandy_loam", 1.45, 0.55, 3.0, "simulated_prototype", "provisional_constant")


def longwall_priors(context: GeologyContext) -> LongwallParameterPriors:
    """Map the simulated context to replaceable PIM/Knothe prototype priors."""
    if context.source != "simulated_prototype":
        raise ValueError("real geology prior mapping is not implemented yet")
    return LongwallParameterPriors(0.70, 55.0, 0.015, 100.0, "simulated_prototype")


def explain_simulated_priors(latitude: float, longitude: float) -> dict:
    context = simulated_geology_context(latitude, longitude)
    return {"context": asdict(context), "longwall_priors": asdict(longwall_priors(context))}

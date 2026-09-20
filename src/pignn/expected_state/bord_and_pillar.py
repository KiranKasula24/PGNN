"""Deterministic Bord-and-Pillar structural expected-state calculations.

This module deliberately contains no learned correction.  CPHSR is a visibly
provisional score until source-paper coefficients are supplied.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CPHSRParameters:
    rock_to_soil_ratio: float
    depth_m: float
    extraction_height_m: float
    brittleness_index: float
    rock_density_kg_m3: float


@dataclass(frozen=True)
class CPHSRResult:
    score_0_to_1: float
    coefficients_verified: bool = False
    weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)


@dataclass(frozen=True)
class Pillar:
    pillar_id: str
    width_m: float
    gallery_width_m: float
    adjacent_pillar_ids: tuple[str, ...]


@dataclass(frozen=True)
class PillarState:
    pillar_id: str
    stress_mpa: float
    strength_mpa: float
    safety_factor: float
    failed: bool


@dataclass(frozen=True)
class BordAndPillarMonitoringResult:
    cphsr: CPHSRResult
    pillars: list[PillarState]
    combined_structural_risk_0_to_1: float
    is_hypothetical: bool = False


def _min_max(value: float, low: float, high: float) -> float:
    if high <= low:
        raise ValueError("invalid normalization range")
    return min(1.0, max(0.0, (value - low) / (high - low)))


def cphsr_score(params: CPHSRParameters) -> CPHSRResult:
    """Provisional equal-weight CPHSR-shaped score, not validated CPHSR.

    Higher depth/height, brittleness, density and lower rock-to-soil ratio are
    treated as higher susceptibility only as a replaceable prototype policy.
    """
    if min(params.rock_to_soil_ratio, params.depth_m, params.extraction_height_m, params.rock_density_kg_m3) <= 0:
        raise ValueError("CPHSR physical inputs must be positive")
    ratio_risk = 1.0 - _min_max(params.rock_to_soil_ratio, 0.0, 4.0)
    depth_height_risk = _min_max(params.depth_m / params.extraction_height_m, 20.0, 100.0)
    brittleness_risk = _min_max(params.brittleness_index, 0.0, 1.0)
    density_risk = _min_max(params.rock_density_kg_m3, 1800.0, 3000.0)
    return CPHSRResult(0.25 * (ratio_risk + depth_height_risk + brittleness_risk + density_risk))


def safety_factor(pillar_width_m: float, gallery_width_m: float, depth_m: float, intact_strength_mpa: float, unit_weight_mpa_per_m: float) -> PillarState:
    """Use tributary-area stress and Obert-Duvall/CMRI pillar strength exactly."""
    if min(pillar_width_m, gallery_width_m, depth_m, intact_strength_mpa, unit_weight_mpa_per_m) <= 0:
        raise ValueError("pillar dimensions, strength, and unit weight must be positive")
    stress = unit_weight_mpa_per_m * depth_m * ((pillar_width_m + gallery_width_m) / pillar_width_m) ** 2
    strength = intact_strength_mpa * (0.778 + 0.222 * pillar_width_m / depth_m)
    return PillarState("", stress, strength, strength / stress, strength / stress < 1.0)


def structural_risk_from_safety_factor(safety_factor_value: float, safe_safety_factor: float = 1.5) -> float:
    """Map failure margin to risk; safe pillars must not retain raw 1/SF risk."""
    if safety_factor_value < 0 or safe_safety_factor <= 1:
        raise ValueError("safety factors must be non-negative and safe threshold must exceed one")
    if safety_factor_value <= 1:
        return 1.0
    if safety_factor_value >= safe_safety_factor:
        return 0.0
    return 1.0 - (safety_factor_value - 1.0) / (safe_safety_factor - 1.0)


def cascading_redistribution(pillars: Iterable[Pillar], depth_m: float, intact_strength_mpa: float, unit_weight_mpa_per_m: float, max_iterations: int = 100) -> list[PillarState]:
    """Redistribute a failed pillar's tributary load over its nonfailed neighbours."""
    by_id = {pillar.pillar_id: pillar for pillar in pillars}
    if not by_id:
        raise ValueError("at least one pillar is required")
    states = {
        pillar_id: safety_factor(pillar.width_m, pillar.gallery_width_m, depth_m, intact_strength_mpa, unit_weight_mpa_per_m)
        for pillar_id, pillar in by_id.items()
    }
    stresses = {pillar_id: state.stress_mpa for pillar_id, state in states.items()}
    failed: set[str] = set()
    for _ in range(max_iterations):
        newly_failed = [pillar_id for pillar_id, state in states.items() if state.safety_factor < 1.0 and pillar_id not in failed]
        if not newly_failed:
            return [PillarState(pillar_id, stresses[pillar_id], states[pillar_id].strength_mpa, states[pillar_id].strength_mpa / max(stresses[pillar_id], 1e-12), pillar_id in failed) for pillar_id in by_id]
        for pillar_id in newly_failed:
            failed.add(pillar_id)
            neighbours = [neighbour for neighbour in by_id[pillar_id].adjacent_pillar_ids if neighbour in by_id and neighbour not in failed]
            if not neighbours:
                continue
            transferred_load = stresses[pillar_id] / len(neighbours)
            stresses[pillar_id] = 0.0
            for neighbour in neighbours:
                stresses[neighbour] += transferred_load
        states = {
            pillar_id: PillarState(pillar_id, stresses[pillar_id], state.strength_mpa, state.strength_mpa / max(stresses[pillar_id], 1e-12), pillar_id in failed)
            for pillar_id, state in states.items()
        }
    raise RuntimeError("pillar load redistribution did not converge")


class BordAndPillarExpectedStateEngine:
    """Semi-static structural state; it does not produce Longwall-style mm fields."""
    def monitoring_mode(self, params: CPHSRParameters, pillars: Iterable[Pillar], depth_m: float, intact_strength_mpa: float, unit_weight_mpa_per_m: float) -> BordAndPillarMonitoringResult:
        cphsr = cphsr_score(params)
        states = cascading_redistribution(pillars, depth_m, intact_strength_mpa, unit_weight_mpa_per_m)
        pillar_risk = max((structural_risk_from_safety_factor(state.safety_factor) for state in states), default=0.0)
        return BordAndPillarMonitoringResult(cphsr, states, max(cphsr.score_0_to_1, pillar_risk))

    def planning_mode(self, params: CPHSRParameters, pillars: Iterable[Pillar], depth_m: float, intact_strength_mpa: float, unit_weight_mpa_per_m: float, hypothetical_extraction_percent: float) -> BordAndPillarMonitoringResult:
        if not 0 <= hypothetical_extraction_percent <= 1:
            raise ValueError("hypothetical_extraction_percent must be in [0, 1]")
        result = self.monitoring_mode(params, pillars, depth_m, intact_strength_mpa, unit_weight_mpa_per_m)
        # Temporary linear extraction-state projection; replace when CPHSR's
        # verified mining-state relationship is available.
        projected_risk = min(1.0, result.combined_structural_risk_0_to_1 * hypothetical_extraction_percent / 0.55)
        return BordAndPillarMonitoringResult(result.cphsr, result.pillars, projected_risk, is_hypothetical=True)

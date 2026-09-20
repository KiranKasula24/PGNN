"""Mine-type-agnostic, two-signal cloud-side Risk Synthesis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ConfidenceBadge = Literal["agreeing", "mixed", "disagreeing"]
RecommendedAction = Literal["monitor", "inspect", "restrict_access", "emergency_review"]


@dataclass(frozen=True)
class RiskSynthesisResult:
    risk_score: float
    recommended_action: RecommendedAction
    confidence_badge: ConfidenceBadge
    low_days: float | None = None
    high_days: float | None = None


def _bounded(value: float, name: str) -> float:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return float(value)


def synthesize(risk_input_a: float, risk_input_b: float, low_days: float | None = None, high_days: float | None = None) -> RiskSynthesisResult:
    """Fuse exactly two normalized signals without letting either permanently veto.

    The caller supplies GNN severity plus Physics Deviation Index for Longwall,
    or GP residual plus structural risk for Bord-and-Pillar.  Agreement is
    deliberately returned separately and never enters the risk calculation.
    """
    a, b = _bounded(risk_input_a, "risk_input_a"), _bounded(risk_input_b, "risk_input_b")
    if low_days is not None and low_days < 0 or high_days is not None and high_days < 0:
        raise ValueError("time-to-threshold bounds must be non-negative")
    if low_days is not None and high_days is not None and low_days > high_days:
        raise ValueError("low_days cannot exceed high_days")
    risk_score = 1 - (1 - a) * (1 - b)
    difference = abs(a - b)
    confidence_badge: ConfidenceBadge = "agreeing" if difference <= 0.20 else "mixed" if difference <= 0.50 else "disagreeing"
    recommended_action: RecommendedAction
    if risk_score < 0.30:
        recommended_action = "monitor"
    elif risk_score < 0.60:
        recommended_action = "inspect"
    elif risk_score < 0.80:
        recommended_action = "restrict_access"
    else:
        recommended_action = "emergency_review"
    return RiskSynthesisResult(risk_score, recommended_action, confidence_badge, low_days, high_days)

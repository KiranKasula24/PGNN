"""One-time GNSS-registration baseline quantities for expected-state comparison."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class NodeBaseline:
    node_id: int
    baseline_displacement_mm: float
    installed_at: datetime


def cumulative_displacement_since_baseline(current_displacement_mm: float, baseline: NodeBaseline) -> float:
    """Return the only observed displacement quantity comparable to Knothe output."""
    return float(current_displacement_mm) - baseline.baseline_displacement_mm

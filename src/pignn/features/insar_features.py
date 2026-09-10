"""Pre-processed InSAR numeric-field validation; deliberately no raster handling."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class InSARSample:
    los_displacement_or_velocity_mm: float | None
    coherence: float | None
    coherence_threshold: float = 0.5

    def validated(self) -> tuple[float, bool]:
        """Low/unknown coherence is no-data, never a zero-displacement reading."""
        if self.coherence is None or self.los_displacement_or_velocity_mm is None:
            return 0.0, False
        if not 0 <= self.coherence <= 1 or not math.isfinite(self.los_displacement_or_velocity_mm):
            return 0.0, False
        return (self.los_displacement_or_velocity_mm, self.coherence >= self.coherence_threshold)

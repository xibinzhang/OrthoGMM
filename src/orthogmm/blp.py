from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .types import Array


@dataclass(frozen=True, slots=True)
class FidelityConfig:
    """Complete numerical configuration for one BLP fidelity level."""

    name: str
    draws: int
    contraction_tolerance: float
    max_iterations: int
    seed: int = 0
    options: dict[str, Any] = field(default_factory=dict)


class MultiFidelityBLPModel(ABC):
    """Abstract adapter implementing Appendix C's BLP partition.

    Subclasses supply market-level moments at low and high fidelity. The
    demanding moments are defined canonically as high minus low.
    """

    low_fidelity: FidelityConfig
    high_fidelity: FidelityConfig

    @abstractmethod
    def moments_at_fidelity(self, theta: Array, fidelity: FidelityConfig) -> Array:
        """Return market-by-moment contributions at the requested fidelity."""

    def tractable_moments(self, theta: Array) -> Array:
        return np.asarray(self.moments_at_fidelity(theta, self.low_fidelity), dtype=float)

    def demanding_moments(self, theta: Array) -> Array:
        high = np.asarray(self.moments_at_fidelity(theta, self.high_fidelity), dtype=float)
        low = np.asarray(self.moments_at_fidelity(theta, self.low_fidelity), dtype=float)
        return high - low

    def reconstruct(self, theta: Array) -> Any:
        """Override to compute high-fidelity elasticities or counterfactuals."""
        return None

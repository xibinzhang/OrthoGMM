from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Protocol

from ..types import Array, EvaluationCounts


MomentKind = Literal["tractable", "demanding"]


class MomentModelProtocol(Protocol):
    """Minimal model contract required by Jacobian strategies."""

    def tractable_moments(self, theta: Array) -> Array:
        ...

    def demanding_moments(self, theta: Array) -> Array:
        ...


@dataclass(slots=True)
class JacobianContext:
    """Information required to evaluate and account for a Jacobian."""

    model: MomentModelProtocol
    theta: Array
    moment_kind: MomentKind
    counts: EvaluationCounts
    bounds: list[tuple[float | None, float | None]] | None = None


class JacobianStrategy(ABC):
    """Common interface for Jacobian construction strategies."""

    @abstractmethod
    def compute(self, context: JacobianContext) -> Array:
        """Return a moment-by-parameter Jacobian matrix."""

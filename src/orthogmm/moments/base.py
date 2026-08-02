"""Abstract contracts for constructing unit-level moment data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .types import MomentData


class MomentBuilder(ABC):
    """Construct a validated :class:`MomentData` representation."""

    @abstractmethod
    def build(self, source: Any, **kwargs: Any) -> MomentData:
        """Build unit-level moments from an external result object."""

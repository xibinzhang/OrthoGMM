from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..types import Array, GMMResult, MomentModel


class BaseEstimator(ABC):
    """Common estimator interface used by all OrthoGMM estimators."""

    result_: GMMResult | None = None

    @abstractmethod
    def fit(self, model: MomentModel, theta0: Array, **kwargs: Any) -> GMMResult:
        """Fit an estimator and return a :class:`GMMResult`."""

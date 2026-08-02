"""Public-API PyBLP moment representation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import MomentBuilder
from .types import MomentData


class PyBLPMomentBuilder(MomentBuilder):
    """Validate PyBLP moment metadata and attach unit contributions.

    PyBLP publicly exposes the aggregate moment vector, its Jacobian,
    covariance matrix, and weighting matrix, but not a market-by-moment
    contribution matrix. Therefore this builder does not fabricate unit
    contributions. Callers must supply a rigorously constructed
    ``market_moments`` matrix.

    The next implementation milestone will construct that matrix block by
    block from aggregate IV residuals and micro-moment pseudo-contributions.
    """

    def build(
        self,
        source: Any,
        *,
        market_ids: np.ndarray,
        market_moments: np.ndarray,
    ) -> MomentData:
        """Combine public PyBLP metadata with market-level contributions."""

        average = self._vector(source, "moments")
        jacobian = self._matrix(source, "moments_jacobian")
        weighting = self._matrix(source, "W")
        covariance = self._matrix(
            source,
            "moments_covariances",
        )

        market_ids = np.asarray(market_ids)
        market_moments = np.asarray(
            market_moments,
            dtype=float,
        )

        if market_moments.ndim != 2:
            raise ValueError(
                "market_moments must be a market-by-moment matrix."
            )

        if market_moments.shape[1] != average.size:
            raise ValueError(
                "market_moments has an incompatible number of moments: "
                f"expected {average.size}, got "
                f"{market_moments.shape[1]}."
            )

        if jacobian.shape[0] != average.size:
            raise ValueError(
                "moments_jacobian has an incompatible number of rows."
            )

        if weighting.shape != (average.size, average.size):
            raise ValueError(
                "W has an incompatible shape."
            )

        if covariance.shape != (average.size, average.size):
            raise ValueError(
                "moments_covariances has an incompatible shape."
            )

        return MomentData(
            unit_ids=market_ids,
            unit_moments=market_moments,
            average_moments=average,
            jacobian=jacobian,
            weighting=weighting,
            covariance=covariance,
        )

    @staticmethod
    def metadata(source: Any) -> dict[str, int]:
        """Return dimensions exposed by the public PyBLP results API."""

        moments = PyBLPMomentBuilder._vector(
            source,
            "moments",
        )
        jacobian = PyBLPMomentBuilder._matrix(
            source,
            "moments_jacobian",
        )

        if jacobian.shape[0] != moments.size:
            raise ValueError(
                "moments and moments_jacobian are dimensionally "
                "incompatible."
            )

        return {
            "n_moments": int(moments.size),
            "n_parameters": int(jacobian.shape[1]),
        }

    @staticmethod
    def _vector(source: Any, name: str) -> np.ndarray:
        if not hasattr(source, name):
            raise AttributeError(
                f"PyBLP results do not expose {name!r}."
            )

        value = np.asarray(
            getattr(source, name),
            dtype=float,
        ).reshape(-1)

        if not np.all(np.isfinite(value)):
            raise ValueError(
                f"PyBLP results attribute {name!r} contains "
                "non-finite values."
            )

        return value

    @staticmethod
    def _matrix(source: Any, name: str) -> np.ndarray:
        if not hasattr(source, name):
            raise AttributeError(
                f"PyBLP results do not expose {name!r}."
            )

        value = np.asarray(
            getattr(source, name),
            dtype=float,
        )

        if value.ndim != 2:
            raise ValueError(
                f"PyBLP results attribute {name!r} must be "
                "two-dimensional."
            )

        if not np.all(np.isfinite(value)):
            raise ValueError(
                f"PyBLP results attribute {name!r} contains "
                "non-finite values."
            )

        return value

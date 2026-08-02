"""Validated moment representations for OrthoGMM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class MomentData:
    """Moment objects aligned at the statistical-unit level.

    Parameters
    ----------
    unit_ids
        One identifier for each statistical unit, such as a market.
    unit_moments
        Unit-by-moment contribution matrix.
    average_moments
        Sample average of the unit contributions.
    jacobian
        Jacobian of the average moment vector.
    weighting
        GMM weighting matrix.
    covariance
        Optional covariance matrix of the moment vector.
    """

    unit_ids: NDArray
    unit_moments: FloatArray
    average_moments: FloatArray
    jacobian: FloatArray
    weighting: FloatArray
    covariance: FloatArray | None = None

    def __post_init__(self) -> None:
        unit_ids = np.asarray(self.unit_ids)
        unit_moments = np.asarray(self.unit_moments, dtype=float)
        average_moments = np.asarray(self.average_moments, dtype=float)
        jacobian = np.asarray(self.jacobian, dtype=float)
        weighting = np.asarray(self.weighting, dtype=float)
        covariance = (
            None
            if self.covariance is None
            else np.asarray(self.covariance, dtype=float)
        )

        if unit_ids.ndim != 1:
            raise ValueError("unit_ids must be one-dimensional.")

        if unit_moments.ndim != 2:
            raise ValueError(
                "unit_moments must be a unit-by-moment matrix."
            )

        n_units, n_moments = unit_moments.shape

        if n_units == 0 or n_moments == 0:
            raise ValueError(
                "unit_moments must have positive dimensions."
            )

        if unit_ids.shape[0] != n_units:
            raise ValueError(
                "unit_ids and unit_moments must contain the same "
                "number of statistical units."
            )

        if np.unique(unit_ids).size != n_units:
            raise ValueError("unit_ids must be unique.")

        if average_moments.shape != (n_moments,):
            raise ValueError(
                f"average_moments must have shape ({n_moments},)."
            )

        if jacobian.ndim != 2 or jacobian.shape[0] != n_moments:
            raise ValueError(
                "jacobian must have one row per moment."
            )

        if weighting.shape != (n_moments, n_moments):
            raise ValueError(
                "weighting must be square with one row and column "
                "per moment."
            )

        if covariance is not None and covariance.shape != (
            n_moments,
            n_moments,
        ):
            raise ValueError(
                "covariance must be square with one row and column "
                "per moment."
            )

        arrays = (
            unit_moments,
            average_moments,
            jacobian,
            weighting,
        )
        if covariance is not None:
            arrays = arrays + (covariance,)

        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("MomentData contains non-finite values.")

        implied_average = unit_moments.mean(axis=0)
        if not np.allclose(
            implied_average,
            average_moments,
            rtol=1e-8,
            atol=1e-10,
        ):
            maximum = float(
                np.max(np.abs(implied_average - average_moments))
            )
            raise ValueError(
                "average_moments do not equal the mean of "
                "unit_moments; maximum absolute discrepancy is "
                f"{maximum:.3e}."
            )

        object.__setattr__(self, "unit_ids", unit_ids)
        object.__setattr__(self, "unit_moments", unit_moments)
        object.__setattr__(self, "average_moments", average_moments)
        object.__setattr__(self, "jacobian", jacobian)
        object.__setattr__(self, "weighting", weighting)
        object.__setattr__(self, "covariance", covariance)

    @property
    def n_units(self) -> int:
        return int(self.unit_moments.shape[0])

    @property
    def n_moments(self) -> int:
        return int(self.unit_moments.shape[1])

    @property
    def n_parameters(self) -> int:
        return int(self.jacobian.shape[1])

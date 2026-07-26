"""
Monte Carlo data-generating designs for OrthoGMM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass
class LinearIVDesign:
    """
    Linear instrumental-variables Monte Carlo design.

    The data-generating process is

        y = X beta + u,

    where one regressor is endogenous and the instruments are correlated
    with the endogenous regressor.

    Parameters
    ----------
    n
        Number of observations.
    beta
        True coefficient vector.
    n_instruments
        Number of excluded instruments.
    endogeneity
        Correlation between the structural error and the endogenous
        component of the regressor.
    instrument_strength
        Strength of the relationship between instruments and the
        endogenous regressor.
    sigma
        Standard deviation of the structural error.
    seed
        Random-number seed.
    """

    n: int = 1000
    beta: tuple[float, ...] = (1.0, -0.5)
    n_instruments: int = 5
    endogeneity: float = 0.5
    instrument_strength: float = 0.8
    sigma: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError("n must be positive.")

        if len(self.beta) != 2:
            raise ValueError(
                "The initial LinearIVDesign requires exactly two coefficients."
            )

        if self.n_instruments <= 0:
            raise ValueError("n_instruments must be positive.")

        if not -0.999 < self.endogeneity < 0.999:
            raise ValueError("endogeneity must lie strictly between -0.999 and 0.999.")

        if self.instrument_strength <= 0:
            raise ValueError("instrument_strength must be positive.")

        if self.sigma <= 0:
            raise ValueError("sigma must be positive.")

    def generate(self, seed: int | None = None) -> dict[str, Any]:
        """
        Generate one simulated data set.

        Parameters
        ----------
        seed
            Optional replication-specific seed. If omitted, the design seed
            is used.

        Returns
        -------
        dict
            Dictionary containing ``y``, ``X``, ``Z``, ``theta_true``,
            and basic design metadata.
        """

        effective_seed = self.seed if seed is None else seed
        rng = np.random.default_rng(effective_seed)

        instruments = rng.normal(
            size=(self.n, self.n_instruments)
        )

        exogenous_regressor = rng.normal(size=self.n)

        structural_shock = rng.normal(size=self.n)
        independent_shock = rng.normal(size=self.n)

        endogenous_disturbance = (
            self.endogeneity * structural_shock
            + np.sqrt(1.0 - self.endogeneity**2) * independent_shock
        )

        instrument_index = instruments.mean(axis=1)

        endogenous_regressor = (
            self.instrument_strength * instrument_index
            + 0.5 * exogenous_regressor
            + endogenous_disturbance
        )

        X = np.column_stack(
            (
                exogenous_regressor,
                endogenous_regressor,
            )
        ).astype(float)

        intercept = np.ones((self.n, 1), dtype=float)

        Z = np.column_stack(
            (
                intercept,
                exogenous_regressor,
                instruments,
            )
        ).astype(float)

        theta_true = np.asarray(self.beta, dtype=float)

        structural_error = self.sigma * structural_shock
        y = X @ theta_true + structural_error

        return {
            "y": y.astype(float),
            "X": X,
            "Z": Z,
            "theta_true": theta_true,
            "n": self.n,
            "seed": effective_seed,
        }
    

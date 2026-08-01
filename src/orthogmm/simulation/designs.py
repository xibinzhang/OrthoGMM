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
            raise ValueError(
                "endogeneity must lie strictly between -0.999 and 0.999."
            )
        if self.instrument_strength <= 0:
            raise ValueError("instrument_strength must be positive.")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive.")

    def generate(self, seed: int | None = None) -> dict[str, Any]:
        effective_seed = self.seed if seed is None else seed
        rng = np.random.default_rng(effective_seed)

        instruments = rng.normal(size=(self.n, self.n_instruments))
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
            (exogenous_regressor, endogenous_regressor)
        ).astype(float)
        intercept = np.ones((self.n, 1), dtype=float)
        Z = np.column_stack(
            (intercept, exogenous_regressor, instruments)
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


@dataclass
class NonlinearRandomCoefficientDesign:
    """
    Nonlinear random-coefficient Monte Carlo design from Section 5.

    The data-generating process is

        Y = alpha + beta X + X A + sigma epsilon,

    where

        A = exp(lambda U) - exp(lambda**2 / 2),

    and X, U, and epsilon are mutually independent standard normal
    random variables.
    """

    n: int = 500
    alpha: float = 0.0
    beta: float = 1.0
    lambda_: float = 0.5
    sigma: float = 0.8
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError("n must be positive.")
        if not np.isfinite(self.alpha):
            raise ValueError("alpha must be finite.")
        if not np.isfinite(self.beta):
            raise ValueError("beta must be finite.")
        if not np.isfinite(self.lambda_):
            raise ValueError("lambda_ must be finite.")
        if self.lambda_ < 0:
            raise ValueError("lambda_ must be non-negative.")
        if not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("sigma must be positive and finite.")

    @property
    def theta_true(self) -> FloatArray:
        return np.asarray([self.alpha, self.beta], dtype=float)

    def generate(self, seed: int | None = None) -> dict[str, Any]:
        effective_seed = self.seed if seed is None else seed
        rng = np.random.default_rng(effective_seed)

        x = rng.normal(size=self.n)
        u = rng.normal(size=self.n)
        epsilon = rng.normal(size=self.n)

        centring_term = np.exp(0.5 * self.lambda_**2)
        random_coefficient = (
            np.exp(self.lambda_ * u) - centring_term
        )
        y = (
            self.alpha
            + self.beta * x
            + x * random_coefficient
            + self.sigma * epsilon
        )

        return {
            "x": x.astype(float),
            "y": y.astype(float),
            "u": u.astype(float),
            "epsilon": epsilon.astype(float),
            "random_coefficient": random_coefficient.astype(float),
            "theta_true": self.theta_true,
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "lambda": float(self.lambda_),
            "sigma": float(self.sigma),
            "n": int(self.n),
            "seed": effective_seed,
        }

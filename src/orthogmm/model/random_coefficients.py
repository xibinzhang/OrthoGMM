from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Literal

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import eval_hermitenorm, logsumexp, ndtr

from ..exceptions import ModelContractError
from ..types import Array
from .base import BaseMomentModel


IntegrationMode = Literal["quadrature", "simulation"]
BasisFamily = Literal["hermite", "fourier"]


@dataclass(frozen=True, slots=True)
class RandomCoefficientIntegration:
    """Numerical integration settings for the demanding score moments."""

    mode: IntegrationMode = "quadrature"
    q_nodes: int = 40
    simulation_draws: int = 250
    seed: int = 20260712

    def __post_init__(self) -> None:
        if self.mode not in ("quadrature", "simulation"):
            raise ValueError(
                "mode must be 'quadrature' or 'simulation'."
            )
        if self.q_nodes <= 0:
            raise ValueError("q_nodes must be positive.")
        if self.simulation_draws <= 0:
            raise ValueError(
                "simulation_draws must be positive."
            )


def normalized_hermite_basis(
    x: Array,
    k: int,
    *,
    start_order: int = 0,
) -> Array:
    """Return normalized probabilists' Hermite basis functions."""

    if k < 1:
        raise ValueError("k must be at least 1.")
    if start_order < 0:
        raise ValueError("start_order must be non-negative.")

    x = np.asarray(x, dtype=float)

    return np.column_stack(
        [
            eval_hermitenorm(order, x)
            / np.sqrt(float(factorial(order)))
            for order in range(
                start_order,
                start_order + k,
            )
        ]
    )


def normalized_fourier_basis(
    x: Array,
    k: int,
) -> Array:
    """Return bounded Fourier features after the transform Phi(X)."""

    if k < 1:
        raise ValueError("k must be at least 1.")

    x = np.asarray(x, dtype=float)
    v = np.clip(
        ndtr(x),
        1e-12,
        1.0 - 1e-12,
    )

    columns: list[Array] = []

    for j in range(k):
        frequency = j // 2 + 1
        angle = 2.0 * np.pi * frequency * v

        if j % 2 == 0:
            columns.append(
                np.sqrt(2.0) * np.cos(angle)
            )
        else:
            columns.append(
                np.sqrt(2.0) * np.sin(angle)
            )

    return np.column_stack(columns)


class RandomCoefficientMomentModel(BaseMomentModel):
    """
    Moment model for the nonlinear random-coefficient design in Section 5.

    The tractable moments are

        g_i(theta) = (e_i(theta), x_i e_i(theta))',

    where e_i(theta) = y_i - alpha - beta x_i.

    The demanding moments multiply the integrated likelihood score by a
    Hermite or Fourier basis in x. The heterogeneity parameter lambda and
    additive-error scale sigma are treated as known design quantities, as
    in the current Section 5 experiment.
    """

    def __init__(
        self,
        x: Array,
        y: Array,
        *,
        lambda_: float = 0.5,
        sigma: float = 0.8,
        basis_k: int = 1,
        basis_start_order: int = 0,
        basis_family: BasisFamily = "fourier",
        integration: RandomCoefficientIntegration | None = None,
    ) -> None:
        self.x = np.asarray(x, dtype=float).reshape(-1)
        self.y = np.asarray(y, dtype=float).reshape(-1)

        if self.x.size == 0:
            raise ModelContractError(
                "x and y must contain at least one observation."
            )
        if self.x.shape != self.y.shape:
            raise ModelContractError(
                "x and y must have the same shape."
            )
        if not np.all(np.isfinite(self.x)):
            raise ModelContractError(
                "x contains non-finite values."
            )
        if not np.all(np.isfinite(self.y)):
            raise ModelContractError(
                "y contains non-finite values."
            )
        if not np.isfinite(lambda_) or lambda_ < 0:
            raise ValueError(
                "lambda_ must be non-negative and finite."
            )
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(
                "sigma must be positive and finite."
            )
        if basis_k < 1:
            raise ValueError("basis_k must be at least 1.")
        if basis_start_order < 0:
            raise ValueError(
                "basis_start_order must be non-negative."
            )
        if basis_family not in ("hermite", "fourier"):
            raise ValueError(
                "basis_family must be 'hermite' or 'fourier'."
            )

        self.lambda_ = float(lambda_)
        self.sigma = float(sigma)
        self.basis_k = int(basis_k)
        self.basis_start_order = int(
            basis_start_order
        )
        self.basis_family = basis_family
        self.integration = (
            integration
            if integration is not None
            else RandomCoefficientIntegration()
        )

        if basis_family == "hermite":
            self.basis = normalized_hermite_basis(
                self.x,
                self.basis_k,
                start_order=self.basis_start_order,
            )
        else:
            self.basis = normalized_fourier_basis(
                self.x,
                self.basis_k,
            )

        self._integration_nodes, self._log_weights = (
            self._build_integration_rule()
        )

    @classmethod
    def from_data(
        cls,
        data: dict,
        **kwargs,
    ) -> "RandomCoefficientMomentModel":
        """Construct the model from a simulation-design data dictionary."""

        required = ("x", "y", "lambda", "sigma")
        missing = [
            name for name in required if name not in data
        ]

        if missing:
            raise ModelContractError(
                "Simulation data are missing: "
                + ", ".join(missing)
                + "."
            )

        return cls(
            data["x"],
            data["y"],
            lambda_=float(data["lambda"]),
            sigma=float(data["sigma"]),
            **kwargs,
        )

    @property
    def demanding_dimension(self) -> int:
        """Number of demanding moment conditions."""

        return 2 * self.basis.shape[1]

    def tractable_moments(
        self,
        theta: Array,
    ) -> Array:
        theta = self._validate_theta(theta)
        alpha, beta = theta
        residual = self.y - alpha - beta * self.x

        return np.column_stack(
            (
                residual,
                self.x * residual,
            )
        )

    def demanding_moments(
        self,
        theta: Array,
    ) -> Array:
        score = self._integrated_score(theta)

        return np.einsum(
            "nk,np->nkp",
            self.basis,
            score,
        ).reshape(
            self.x.size,
            self.demanding_dimension,
        )

    def tractable_jacobian(
        self,
        theta: Array,
    ) -> Array:
        self._validate_theta(theta)

        return np.asarray(
            [
                [-1.0, -np.mean(self.x)],
                [-np.mean(self.x), -np.mean(self.x**2)],
            ],
            dtype=float,
        )

    def _integrated_score(
        self,
        theta: Array,
    ) -> Array:
        theta = self._validate_theta(theta)
        alpha, beta = theta

        u = self._integration_nodes[None, :]
        x = self.x[:, None]
        y = self.y[:, None]

        random_coefficient = (
            np.exp(
                np.clip(
                    self.lambda_ * u,
                    -40.0,
                    40.0,
                )
            )
            - np.exp(0.5 * self.lambda_**2)
        )

        residual = (
            y
            - alpha
            - beta * x
            - x * random_coefficient
        )
        standardized = residual / self.sigma

        log_component = (
            self._log_weights[None, :]
            - np.log(self.sigma)
            - 0.5 * np.log(2.0 * np.pi)
            - 0.5 * standardized**2
        )

        posterior = np.exp(
            log_component
            - logsumexp(
                log_component,
                axis=1,
                keepdims=True,
            )
        )

        score_alpha = residual / self.sigma**2
        score_beta = x * residual / self.sigma**2
        component_scores = np.stack(
            (score_alpha, score_beta),
            axis=2,
        )

        integrated_score = np.sum(
            posterior[:, :, None] * component_scores,
            axis=1,
        )

        if not np.all(np.isfinite(integrated_score)):
            raise ModelContractError(
                "Integrated score contains non-finite values."
            )

        return integrated_score

    def _build_integration_rule(
        self,
    ) -> tuple[Array, Array]:
        if self.integration.mode == "quadrature":
            nodes, weights = hermgauss(
                self.integration.q_nodes
            )
            integration_nodes = np.sqrt(2.0) * nodes
            log_weights = (
                np.log(weights)
                - 0.5 * np.log(np.pi)
            )
        else:
            rng = np.random.default_rng(
                self.integration.seed
            )
            integration_nodes = rng.normal(
                size=self.integration.simulation_draws
            )
            log_weights = np.full(
                self.integration.simulation_draws,
                -np.log(
                    self.integration.simulation_draws
                ),
                dtype=float,
            )

        return (
            np.asarray(integration_nodes, dtype=float),
            np.asarray(log_weights, dtype=float),
        )

    @staticmethod
    def _validate_theta(theta: Array) -> Array:
        theta = np.asarray(theta, dtype=float).reshape(-1)

        if theta.shape != (2,):
            raise ModelContractError(
                "theta must contain exactly (alpha, beta)."
            )
        if not np.all(np.isfinite(theta)):
            raise ModelContractError(
                "theta contains non-finite values."
            )

        return theta

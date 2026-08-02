r"""Sequential efficient influence projection estimator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..linalg import solve
from ..operators import (
    ProjectedInformationOperator,
    ProjectedInformationResult,
)
from ..optimization import QuadraticTrustRegion, TrustRegionResult


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SEIPResult:
    """Result of one information-metric SEIP update."""

    theta_initial: FloatArray
    theta_updated: FloatArray
    score: FloatArray
    tractable_score: FloatArray
    demanding_score: FloatArray
    projected_information: ProjectedInformationResult
    trust_region: TrustRegionResult
    retained_rank: int
    explained_variance_ratio: float

    def __post_init__(self) -> None:
        theta_initial = np.asarray(
            self.theta_initial,
            dtype=float,
        ).reshape(-1)
        theta_updated = np.asarray(
            self.theta_updated,
            dtype=float,
        ).reshape(-1)
        score = np.asarray(self.score, dtype=float).reshape(-1)
        tractable_score = np.asarray(
            self.tractable_score,
            dtype=float,
        ).reshape(-1)
        demanding_score = np.asarray(
            self.demanding_score,
            dtype=float,
        ).reshape(-1)

        dimension = theta_initial.size
        arrays = (
            theta_initial,
            theta_updated,
            score,
            tractable_score,
            demanding_score,
        )
        if dimension < 1:
            raise ValueError("The parameter vector cannot be empty.")
        if any(array.size != dimension for array in arrays):
            raise ValueError(
                "All parameter and score vectors must have the same dimension."
            )
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("SEIPResult contains non-finite values.")

        object.__setattr__(self, "theta_initial", theta_initial)
        object.__setattr__(self, "theta_updated", theta_updated)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "tractable_score", tractable_score)
        object.__setattr__(self, "demanding_score", demanding_score)

    @property
    def step(self) -> FloatArray:
        return self.theta_updated - self.theta_initial

    @property
    def step_norm(self) -> float:
        return float(np.linalg.norm(self.step))

    @property
    def score_norm(self) -> float:
        return float(np.linalg.norm(self.score))

    @property
    def relative_step_norm(self) -> float:
        scale = np.maximum(1.0, np.abs(self.theta_initial))
        return float(np.linalg.norm(self.step / scale))


class SequentialEfficientInfluenceProjection:
    r"""Compute one projected GMM update under a trust-region metric.

    The projected score is

    .. math::

       s =
       \widetilde G^\prime
       \Omega_{\widetilde g\widetilde g}^{-1}
       \bar{\widetilde g}
       +
       R^\prime S^{-1}\bar\nu.

    The update solves

    .. math::

       \min_d \frac12 d^\prime Jd + s^\prime d
       \quad\text{subject to}\quad
       d^\prime M d \leq \rho^2.

    By default, ``M = J`` so the trust region is measured in the local
    projected-information metric.
    """

    def __init__(
        self,
        *,
        rank: int | None = None,
        explained_variance: float | None = None,
        singular_value_tolerance: float | None = None,
        ridge: float = 1e-8,
        metric_type: str = "information",
        radius: float | None = 1.0,
        tolerance: float = 1e-10,
        maximum_iterations: int = 100,
        center: bool = True,
    ) -> None:
        if rank is not None and explained_variance is not None:
            raise ValueError(
                "Specify either rank or explained_variance, not both."
            )
        if radius is not None and radius <= 0:
            raise ValueError("radius must be positive or None.")

        self.rank = rank
        self.explained_variance = explained_variance
        self.singular_value_tolerance = singular_value_tolerance
        self.ridge = float(ridge)
        self.metric_type = metric_type
        self.radius = radius
        self.tolerance = float(tolerance)
        self.maximum_iterations = int(maximum_iterations)
        self.center = bool(center)

    def fit(
        self,
        theta: FloatArray,
        g: FloatArray,
        h: FloatArray,
        G: FloatArray,
        H: FloatArray,
    ) -> SEIPResult:
        """Construct projected information and apply one trust-region update."""

        theta_array = np.asarray(theta, dtype=float).reshape(-1)

        projected = ProjectedInformationOperator(
            rank=self.rank,
            explained_variance=self.explained_variance,
            singular_value_tolerance=self.singular_value_tolerance,
            center=self.center,
            ridge=self.ridge,
        ).fit(g, h, G, H)

        if theta_array.size != projected.parameter_dimension:
            raise ValueError(
                "theta dimension does not match the projected information."
            )

        basis = projected.basis_result
        g_bar = basis.reduced_moments.mean(axis=0)
        nu_bar = projected.residual_moments.mean(axis=0)

        tractable_score = basis.reduced_jacobian.T @ solve(
            projected.omega_gg,
            g_bar,
        )
        demanding_score = projected.residual_jacobian.T @ solve(
            projected.schur_complement,
            nu_bar,
        )
        score = tractable_score + demanding_score

        radius = (
            float(np.sqrt(theta_array.size))
            if self.radius is None
            else float(self.radius)
        )

        trust_region = QuadraticTrustRegion(
            radius=radius,
            metric_type=self.metric_type,
            tolerance=self.tolerance,
            maximum_iterations=self.maximum_iterations,
            ridge=self.ridge,
        ).solve(
            projected.information,
            score,
            theta=theta_array,
        )

        return SEIPResult(
            theta_initial=theta_array,
            theta_updated=theta_array + trust_region.step,
            score=score,
            tractable_score=tractable_score,
            demanding_score=demanding_score,
            projected_information=projected,
            trust_region=trust_region,
            retained_rank=basis.retained_rank,
            explained_variance_ratio=basis.explained_variance_ratio,
        )

    __call__ = fit

r"""Quadratic trust-region corrections for projected GMM systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..linalg import solve, stable_matrix


FloatArray = NDArray[np.float64]
MetricType = Literal["euclidean", "parameter_scale", "information"]


@dataclass(frozen=True, slots=True)
class TrustRegionResult:
    """Solution of a local quadratic trust-region problem."""

    step: FloatArray
    unconstrained_step: FloatArray
    lagrange_multiplier: float
    radius: float
    metric_norm: float
    active: bool
    converged: bool
    iterations: int
    objective_value: float
    predicted_reduction: float
    metric: FloatArray
    metric_type: str

    def __post_init__(self) -> None:
        step = np.asarray(self.step, dtype=float).reshape(-1)
        unconstrained = np.asarray(
            self.unconstrained_step,
            dtype=float,
        ).reshape(-1)
        metric = np.asarray(self.metric, dtype=float)

        if step.size < 1 or unconstrained.size != step.size:
            raise ValueError(
                "step and unconstrained_step must have the same "
                "positive dimension."
            )
        if metric.shape != (step.size, step.size):
            raise ValueError("metric has an incompatible shape.")
        if not all(
            np.all(np.isfinite(array))
            for array in (step, unconstrained, metric)
        ):
            raise ValueError(
                "TrustRegionResult contains non-finite values."
            )
        if self.radius <= 0:
            raise ValueError("radius must be positive.")
        if self.lagrange_multiplier < 0:
            raise ValueError(
                "lagrange_multiplier must be nonnegative."
            )

        object.__setattr__(self, "step", step)
        object.__setattr__(
            self,
            "unconstrained_step",
            unconstrained,
        )
        object.__setattr__(self, "metric", metric)

    @property
    def euclidean_norm(self) -> float:
        return float(np.linalg.norm(self.step))

    @property
    def unconstrained_euclidean_norm(self) -> float:
        return float(np.linalg.norm(self.unconstrained_step))


class QuadraticTrustRegion:
    r"""Solve a positive-definite quadratic trust-region problem.

    The problem is

    .. math::

       \min_d \frac12 d^\prime J d + s^\prime d
       \quad\text{subject to}\quad
       d^\prime M d \leq \rho^2,

    where ``J`` is the local projected information matrix and ``M`` is a
    positive-definite trust-region metric. The solution satisfies

    .. math::

       d(\lambda) = -(J + \lambda M)^{-1}s,

    with ``lambda = 0`` for an interior solution and otherwise chosen so
    that the constraint binds.
    """

    def __init__(
        self,
        *,
        radius: float,
        metric_type: MetricType = "parameter_scale",
        tolerance: float = 1e-10,
        maximum_iterations: int = 100,
        ridge: float = 1e-10,
    ) -> None:
        if radius <= 0:
            raise ValueError("radius must be positive.")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive.")
        if maximum_iterations < 1:
            raise ValueError(
                "maximum_iterations must be positive."
            )
        if ridge < 0:
            raise ValueError("ridge must be nonnegative.")
        if metric_type not in {
            "euclidean",
            "parameter_scale",
            "information",
        }:
            raise ValueError("Unknown metric_type.")

        self.radius = float(radius)
        self.metric_type = metric_type
        self.tolerance = float(tolerance)
        self.maximum_iterations = int(maximum_iterations)
        self.ridge = float(ridge)

    def _metric(
        self,
        information: FloatArray,
        theta: FloatArray | None,
    ) -> FloatArray:
        p = information.shape[0]

        if self.metric_type == "euclidean":
            return np.eye(p)

        if self.metric_type == "information":
            return stable_matrix(
                information,
                ridge=self.ridge,
            ).value

        if theta is None:
            raise ValueError(
                "theta is required for parameter_scale metric."
            )
        theta_array = np.asarray(theta, dtype=float).reshape(-1)
        if theta_array.size != p:
            raise ValueError(
                "theta dimension does not match information."
            )
        scales = np.maximum(1.0, np.abs(theta_array))
        return np.diag(1.0 / scales**2)

    @staticmethod
    def _metric_norm(
        step: FloatArray,
        metric: FloatArray,
    ) -> float:
        value = float(step @ metric @ step)
        return float(np.sqrt(max(0.0, value)))

    @staticmethod
    def _objective(
        step: FloatArray,
        information: FloatArray,
        score: FloatArray,
    ) -> float:
        return float(
            0.5 * step @ information @ step
            + score @ step
        )

    def solve(
        self,
        information: FloatArray,
        score: FloatArray,
        *,
        theta: FloatArray | None = None,
        metric: FloatArray | None = None,
    ) -> TrustRegionResult:
        """Solve the trust-region problem."""

        J_raw = np.asarray(information, dtype=float)
        s = np.asarray(score, dtype=float).reshape(-1)

        if J_raw.ndim != 2 or J_raw.shape[0] != J_raw.shape[1]:
            raise ValueError("information must be square.")
        if J_raw.shape[0] != s.size:
            raise ValueError(
                "score dimension does not match information."
            )
        if not np.all(np.isfinite(J_raw)) or not np.all(
            np.isfinite(s)
        ):
            raise ValueError("Inputs contain non-finite values.")

        J = stable_matrix(
            J_raw,
            ridge=self.ridge,
        ).value

        if metric is None:
            M = self._metric(J, theta)
            metric_name = self.metric_type
        else:
            M = stable_matrix(
                np.asarray(metric, dtype=float),
                ridge=self.ridge,
            ).value
            if M.shape != J.shape:
                raise ValueError(
                    "metric dimension does not match information."
                )
            metric_name = "custom"

        unconstrained = -solve(J, s)
        unconstrained_norm = self._metric_norm(
            unconstrained,
            M,
        )

        if unconstrained_norm <= self.radius * (
            1.0 + self.tolerance
        ):
            objective = self._objective(
                unconstrained,
                J,
                s,
            )
            return TrustRegionResult(
                step=unconstrained,
                unconstrained_step=unconstrained,
                lagrange_multiplier=0.0,
                radius=self.radius,
                metric_norm=unconstrained_norm,
                active=False,
                converged=True,
                iterations=0,
                objective_value=objective,
                predicted_reduction=-objective,
                metric=M,
                metric_type=metric_name,
            )

        def step_at(multiplier: float) -> FloatArray:
            return -solve(J + multiplier * M, s)

        lower = 0.0
        upper = 1.0
        upper_step = step_at(upper)

        while (
            self._metric_norm(upper_step, M) > self.radius
            and upper < 1e16
        ):
            upper *= 2.0
            upper_step = step_at(upper)

        if upper >= 1e16 and (
            self._metric_norm(upper_step, M) > self.radius
        ):
            raise RuntimeError(
                "Could not bracket the trust-region multiplier."
            )

        converged = False
        step = upper_step
        multiplier = upper
        iterations = 0

        for iterations in range(1, self.maximum_iterations + 1):
            multiplier = 0.5 * (lower + upper)
            step = step_at(multiplier)
            norm = self._metric_norm(step, M)

            if abs(norm - self.radius) <= (
                self.tolerance * self.radius
            ):
                converged = True
                break

            if norm > self.radius:
                lower = multiplier
            else:
                upper = multiplier

        metric_norm = self._metric_norm(step, M)
        objective = self._objective(step, J, s)

        return TrustRegionResult(
            step=step,
            unconstrained_step=unconstrained,
            lagrange_multiplier=multiplier,
            radius=self.radius,
            metric_norm=metric_norm,
            active=True,
            converged=converged,
            iterations=iterations,
            objective_value=objective,
            predicted_reduction=-objective,
            metric=M,
            metric_type=metric_name,
        )

    __call__ = solve

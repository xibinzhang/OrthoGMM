"""Tractable localization for the Petrin BLP application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .model import PetrinApplicationModel


FloatArray = NDArray[np.float64]


def _scalar_attribute(
    results: Any,
    name: str,
    *,
    default: float = float("nan"),
) -> float:
    value = getattr(results, name, default)
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return default
    return float(array.reshape(-1)[0])


def _integer_attribute(
    results: Any,
    name: str,
    *,
    default: int = 0,
) -> int:
    value = getattr(results, name, default)
    array = np.asarray(value)
    if array.size == 0:
        return default
    return int(array.reshape(-1)[0])


@dataclass(frozen=True, slots=True)
class PetrinLocalizationResult:
    """Result of aggregate-only Petrin optimization."""

    theta_initial: FloatArray
    theta_localized: FloatArray
    objective: float
    projected_gradient_norm: float
    converged: bool
    optimization_iterations: int
    objective_evaluations: int
    fixed_point_iterations: int
    contraction_evaluations: int
    elapsed_seconds: float
    lower_bound_hits: tuple[int, ...]
    upper_bound_hits: tuple[int, ...]
    pyblp_results: Any

    def __post_init__(self) -> None:
        initial = np.asarray(
            self.theta_initial,
            dtype=float,
        ).reshape(-1)
        localized = np.asarray(
            self.theta_localized,
            dtype=float,
        ).reshape(-1)

        if initial.size < 1 or localized.size != initial.size:
            raise ValueError(
                "Initial and localized parameter vectors must have "
                "the same positive dimension."
            )
        if not np.all(np.isfinite(initial)):
            raise ValueError("theta_initial contains non-finite values.")
        if not np.all(np.isfinite(localized)):
            raise ValueError("theta_localized contains non-finite values.")
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be nonnegative.")

        object.__setattr__(self, "theta_initial", initial)
        object.__setattr__(self, "theta_localized", localized)

    @property
    def parameter_dimension(self) -> int:
        return int(self.theta_localized.size)

    @property
    def update(self) -> FloatArray:
        return self.theta_localized - self.theta_initial

    @property
    def update_norm(self) -> float:
        return float(np.linalg.norm(self.update))

    @property
    def relative_update_norm(self) -> float:
        denominator = max(1.0, float(np.linalg.norm(self.theta_initial)))
        return self.update_norm / denominator

    @property
    def on_boundary(self) -> bool:
        return bool(self.lower_bound_hits or self.upper_bound_hits)


class PetrinTractableLocalizer:
    """Optimize the aggregate Petrin criterion without micro moments.

    This delegates nonlinear optimization, parameter bounds, contraction,
    and linear-parameter concentration to PyBLP. The resulting nonlinear
    vector is the preliminary estimator used to construct the local
    demanding block.
    """

    def __init__(
        self,
        model: PetrinApplicationModel,
        *,
        method: str = "1s",
        initial_update: bool | None = None,
        check_optimality: str | None = None,
        boundary_tolerance: float = 1e-8,
    ) -> None:
        if method not in {"1s", "2s"}:
            raise ValueError("method must be '1s' or '2s'.")
        if boundary_tolerance < 0:
            raise ValueError(
                "boundary_tolerance must be nonnegative."
            )

        self.model = model
        self.method = method
        self.initial_update = initial_update
        self.check_optimality = check_optimality
        self.boundary_tolerance = float(boundary_tolerance)

    def _active_bounds(self) -> tuple[FloatArray, FloatArray]:
        """Return active nonlinear bounds when the setup exposes them.

        Some Petrin setup objects are lightweight problem containers and do
        not retain ``sigma_bounds`` and ``pi_bounds`` as public attributes.
        In that case, boundary diagnostics are unavailable, so return
        unbounded vectors rather than failing after a successful PyBLP solve.
        """

        setup = self.model.setup
        mapping = self.model.parameter_map
        dimension = self.model.parameter_dimension

        sigma_bounds = getattr(setup, "sigma_bounds", None)
        pi_bounds = getattr(setup, "pi_bounds", None)

        if sigma_bounds is None or pi_bounds is None:
            return (
                np.full(dimension, -np.inf, dtype=float),
                np.full(dimension, np.inf, dtype=float),
            )

        sigma_lower = np.asarray(sigma_bounds[0], dtype=float)
        sigma_upper = np.asarray(sigma_bounds[1], dtype=float)
        pi_lower = np.asarray(pi_bounds[0], dtype=float)
        pi_upper = np.asarray(pi_bounds[1], dtype=float)

        lower = np.r_[
            sigma_lower.reshape(-1)[mapping.sigma_indices],
            pi_lower.reshape(-1)[mapping.pi_indices],
        ]
        upper = np.r_[
            sigma_upper.reshape(-1)[mapping.sigma_indices],
            pi_upper.reshape(-1)[mapping.pi_indices],
        ]
        return lower, upper

    def fit(
        self,
        theta0: FloatArray | None = None,
    ) -> PetrinLocalizationResult:
        """Run aggregate-only nonlinear optimization."""

        initial = (
            self.model.theta0
            if theta0 is None
            else np.asarray(theta0, dtype=float).reshape(-1)
        )
        sigma, pi = self.model.structural_parameters(initial)

        evaluation = self.model.solver.solve(
            self.model.setup,
            fidelity=self.model.aggregate_fidelity,
            sigma=sigma,
            pi=pi,
            include_micro=False,
            fixed_parameters=False,
            method=self.method,
            initial_update=self.initial_update,
            check_optimality=self.check_optimality,
        )
        results = evaluation.results

        localized = np.asarray(
            results.theta,
            dtype=float,
        ).reshape(-1)
        if localized.size != self.model.parameter_dimension:
            raise ValueError(
                "PyBLP returned a nonlinear parameter vector with an "
                "unexpected dimension."
            )

        lower, upper = self._active_bounds()
        tol = self.boundary_tolerance

        lower_hits = tuple(
            int(index)
            for index in np.flatnonzero(
                np.isfinite(lower)
                & (localized <= lower + tol)
            )
        )
        upper_hits = tuple(
            int(index)
            for index in np.flatnonzero(
                np.isfinite(upper)
                & (localized >= upper - tol)
            )
        )

        converged = bool(
            getattr(
                results,
                "optimization_converged",
                True,
            )
        )

        return PetrinLocalizationResult(
            theta_initial=initial,
            theta_localized=localized,
            objective=_scalar_attribute(results, "objective"),
            projected_gradient_norm=_scalar_attribute(
                results,
                "projected_gradient_norm",
            ),
            converged=converged,
            optimization_iterations=_integer_attribute(
                results,
                "optimization_iterations",
            ),
            objective_evaluations=_integer_attribute(
                results,
                "objective_evaluations",
            ),
            fixed_point_iterations=_integer_attribute(
                results,
                "fixed_point_iterations",
            ),
            contraction_evaluations=_integer_attribute(
                results,
                "contraction_evaluations",
            ),
            elapsed_seconds=evaluation.elapsed_seconds,
            lower_bound_hits=lower_hits,
            upper_bound_hits=upper_hits,
            pyblp_results=results,
        )

    __call__ = fit

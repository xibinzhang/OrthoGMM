"""Micro-assisted localization for the Petrin BLP application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ...blp import FidelityConfig
from .model import PetrinApplicationModel


FloatArray = NDArray[np.float64]


def _scalar(value: Any, default: float = float("nan")) -> float:
    """Extract the first scalar value from a PyBLP result field."""
    if value is None:
        return default
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return default
    return float(array.reshape(-1)[0])


def _integer(value: Any, default: int = 0) -> int:
    """Extract the first integer value from a PyBLP result field."""
    if value is None:
        return default
    array = np.asarray(value)
    if array.size == 0:
        return default
    return int(array.reshape(-1)[0])


def _optimization_stats(results: Any) -> Any | None:
    """Return PyBLP's optimizer statistics object when available."""
    for name in (
        "optimization_stats",
        "_optimization_stats",
        "optimizer_stats",
    ):
        stats = getattr(results, name, None)
        if stats is not None:
            return stats
    return None


def _convergence(results: Any) -> bool:
    """Read optimizer convergence without silently defaulting to success."""
    stats = _optimization_stats(results)
    for owner in (stats, results):
        if owner is None:
            continue
        for name in (
            "converged",
            "optimization_converged",
            "success",
        ):
            value = getattr(owner, name, None)
            if value is not None:
                return bool(np.asarray(value).reshape(-1)[0])
    return False


def _count(results: Any, *names: str) -> int:
    """Read a count from optimizer statistics or the result object."""
    stats = _optimization_stats(results)
    for owner in (stats, results):
        if owner is None:
            continue
        for name in names:
            value = getattr(owner, name, None)
            if value is not None:
                return _integer(value)
    return 0


def _sum_array_attribute(results: Any, name: str) -> int:
    """Sum a PyBLP count array such as fixed-point iterations."""
    value = getattr(results, name, None)
    if value is None:
        return 0
    array = np.asarray(value)
    if array.size == 0:
        return 0
    return int(np.sum(array))


@dataclass(frozen=True, slots=True)
class PetrinMicroLocalizationResult:
    """Result of one aggregate-plus-micro Petrin localization."""

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
    fidelity: FidelityConfig
    pyblp_results: Any

    def __post_init__(self) -> None:
        initial = np.asarray(self.theta_initial, dtype=float).reshape(-1)
        localized = np.asarray(self.theta_localized, dtype=float).reshape(-1)

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
    def update(self) -> FloatArray:
        return self.theta_localized - self.theta_initial

    @property
    def update_norm(self) -> float:
        return float(np.linalg.norm(self.update))

    @property
    def relative_update_norm(self) -> float:
        denominator = max(1.0, float(np.linalg.norm(self.theta_initial)))
        return self.update_norm / denominator


class PetrinMicroLocalizer:
    """Optimize aggregate and micro moments under a coarse fidelity."""

    def __init__(
        self,
        model: PetrinApplicationModel,
        *,
        fidelity: FidelityConfig | None = None,
        method: str = "1s",
        initial_update: bool | None = None,
        check_optimality: str | None = None,
    ) -> None:
        if method not in {"1s", "2s"}:
            raise ValueError("method must be '1s' or '2s'.")

        self.model = model
        self.fidelity = (
            fidelity
            or FidelityConfig(
                name="petrin_micro_localizer",
                draws=model.setup.n_agents,
                contraction_tolerance=1e-6,
                max_iterations=300,
                seed=0,
            )
        )
        self.method = method
        self.initial_update = initial_update
        self.check_optimality = check_optimality

    def fit(
        self,
        theta0: FloatArray | None = None,
    ) -> PetrinMicroLocalizationResult:
        """Run aggregate-plus-micro nonlinear optimization."""

        initial = (
            self.model.theta0
            if theta0 is None
            else np.asarray(theta0, dtype=float).reshape(-1)
        )
        sigma, pi = self.model.structural_parameters(initial)

        evaluation = self.model.solver.solve(
            self.model.setup,
            fidelity=self.fidelity,
            sigma=sigma,
            pi=pi,
            include_micro=True,
            fixed_parameters=False,
            method=self.method,
            initial_update=self.initial_update,
            check_optimality=self.check_optimality,
        )
        results = evaluation.results

        localized = np.asarray(results.theta, dtype=float).reshape(-1)
        if localized.size != self.model.parameter_dimension:
            raise ValueError(
                "PyBLP returned a nonlinear parameter vector with an "
                "unexpected dimension."
            )

        fixed_point_iterations = _sum_array_attribute(
            results, "fixed_point_iterations"
        )
        contraction_evaluations = _sum_array_attribute(
            results, "contraction_evaluations"
        )

        return PetrinMicroLocalizationResult(
            theta_initial=initial,
            theta_localized=localized,
            objective=_scalar(getattr(results, "objective", None)),
            projected_gradient_norm=_scalar(
                getattr(results, "projected_gradient_norm", None)
            ),
            converged=_convergence(results),
            optimization_iterations=_count(
                results, "iterations", "optimization_iterations"
            ),
            objective_evaluations=_count(
                results, "evaluations", "objective_evaluations"
            ),
            fixed_point_iterations=fixed_point_iterations,
            contraction_evaluations=contraction_evaluations,
            elapsed_seconds=evaluation.elapsed_seconds,
            fidelity=self.fidelity,
            pyblp_results=results,
        )

    __call__ = fit

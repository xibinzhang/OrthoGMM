"""Validate saved Petrin updates with fixed-parameter PyBLP solves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ...blp import FidelityConfig
from .model import PetrinApplicationModel


FloatArray = NDArray[np.float64]


def _scalar(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return default
    return float(array.reshape(-1)[0])


def _sum_first_available(results: Any, *names: str) -> int:
    """Sum the first available PyBLP count array.

    Recent PyBLP versions expose cumulative arrays, whereas some older
    versions expose non-cumulative arrays. Prefer the cumulative public
    attributes when present.
    """
    for name in names:
        value = getattr(results, name, None)
        if value is None:
            continue
        array = np.asarray(value)
        if array.size:
            return int(np.sum(array))
    return 0


@dataclass(frozen=True, slots=True)
class PetrinFixedEvaluation:
    """Fixed-parameter PyBLP evaluation at one nonlinear vector."""

    theta: FloatArray
    objective: float
    projected_gradient_norm: float
    fixed_point_iterations: int
    contraction_evaluations: int
    elapsed_seconds: float
    results: Any

    def __post_init__(self) -> None:
        theta = np.asarray(self.theta, dtype=float).reshape(-1)
        if theta.size < 1 or not np.all(np.isfinite(theta)):
            raise ValueError("theta must be a finite nonempty vector.")
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be nonnegative.")
        object.__setattr__(self, "theta", theta)


@dataclass(frozen=True, slots=True)
class PetrinSEIPValidationResult:
    """Comparison of localized and updated fixed evaluations.

    The historical class name is retained for backward compatibility. The
    comparison is generic and is also used for residual-only SOP validation.
    """

    localized: PetrinFixedEvaluation
    updated: PetrinFixedEvaluation

    @property
    def objective_change(self) -> float:
        return self.updated.objective - self.localized.objective

    @property
    def objective_improvement(self) -> float:
        return self.localized.objective - self.updated.objective

    @property
    def relative_objective_improvement(self) -> float:
        denominator = max(abs(self.localized.objective), np.finfo(float).tiny)
        return self.objective_improvement / denominator

    @property
    def objective_improved(self) -> bool:
        return bool(self.updated.objective < self.localized.objective)

    @property
    def step(self) -> FloatArray:
        return self.updated.theta - self.localized.theta

    @property
    def step_norm(self) -> float:
        return float(np.linalg.norm(self.step))


class PetrinSEIPValidator:
    """Validate one saved update using fixed-parameter PyBLP evaluations.

    The historical class name is retained for backward compatibility. No
    nonlinear optimizer is run: both parameter vectors are evaluated with
    ``fixed_parameters=True``.
    """

    def __init__(
        self,
        model: PetrinApplicationModel,
        *,
        fidelity: FidelityConfig | None = None,
        include_micro: bool = True,
        method: str = "1s",
    ) -> None:
        if method not in {"1s", "2s"}:
            raise ValueError("method must be '1s' or '2s'.")

        self.model = model
        if fidelity is not None:
            self.fidelity = fidelity
        else:
            default_fidelity = getattr(model, "high_fidelity", None)
            if default_fidelity is None:
                default_fidelity = getattr(model, "micro_fidelity", None)
            if default_fidelity is None:
                default_fidelity = FidelityConfig(
                    name="petrin_fixed_validation",
                    draws=model.setup.n_agents,
                    contraction_tolerance=1e-12,
                    max_iterations=1000,
                    seed=0,
                )
            self.fidelity = default_fidelity

        self.include_micro = bool(include_micro)
        self.method = method

    def evaluate(self, theta: FloatArray) -> PetrinFixedEvaluation:
        """Evaluate the PyBLP criterion with nonlinear parameters fixed."""

        theta_array = np.asarray(theta, dtype=float).reshape(-1)
        sigma, pi = self.model.structural_parameters(theta_array)

        evaluation = self.model.solver.solve(
            self.model.setup,
            fidelity=self.fidelity,
            sigma=sigma,
            pi=pi,
            include_micro=self.include_micro,
            fixed_parameters=True,
            method=self.method,
        )
        results = evaluation.results

        return PetrinFixedEvaluation(
            theta=theta_array,
            objective=_scalar(getattr(results, "objective", None)),
            projected_gradient_norm=_scalar(
                getattr(results, "projected_gradient_norm", None)
            ),
            fixed_point_iterations=_sum_first_available(
                results,
                "cumulative_fp_iterations",
                "cumulative_fixed_point_iterations",
                "fp_iterations",
                "fixed_point_iterations",
            ),
            contraction_evaluations=_sum_first_available(
                results,
                "cumulative_contraction_evaluations",
                "contraction_evaluations",
            ),
            elapsed_seconds=evaluation.elapsed_seconds,
            results=results,
        )

    def compare(
        self,
        theta_localized: FloatArray,
        theta_updated: FloatArray,
    ) -> PetrinSEIPValidationResult:
        """Evaluate and compare localized and updated parameter vectors."""

        localized = self.evaluate(theta_localized)
        updated = self.evaluate(theta_updated)
        return PetrinSEIPValidationResult(
            localized=localized,
            updated=updated,
        )

    __call__ = compare

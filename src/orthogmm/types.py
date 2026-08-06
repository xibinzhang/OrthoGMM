from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@runtime_checkable
class MomentModel(Protocol):
    """Structural protocol for models accepted by OrthoGMM."""

    def tractable_moments(self, theta: Array) -> Array: ...

    def demanding_moments(self, theta: Array) -> Array: ...


@dataclass(slots=True)
class EvaluationCounts:
    tractable_objective: int = 0
    tractable_moments: int = 0
    demanding_moments_projection: int = 0
    demanding_moments_derivative: int = 0
    tractable_jacobian: int = 0
    demanding_jacobian: int = 0
    failed_evaluations: int = 0
    reconstruction: int = 0

    @property
    def demanding_moments_total(self) -> int:
        return self.demanding_moments_projection + self.demanding_moments_derivative

    def as_dict(self) -> dict[str, int]:
        return {
            "tractable_objective": self.tractable_objective,
            "tractable_moments": self.tractable_moments,
            "demanding_moments_projection": self.demanding_moments_projection,
            "demanding_moments_derivative": self.demanding_moments_derivative,
            "demanding_moments_total": self.demanding_moments_total,
            "tractable_jacobian": self.tractable_jacobian,
            "demanding_jacobian": self.demanding_jacobian,
            "failed_evaluations": self.failed_evaluations,
            "reconstruction": self.reconstruction,
        }


@dataclass(slots=True)
class StageTimings:
    localization: float = 0.0
    moments: float = 0.0
    derivatives: float = 0.0
    projection: float = 0.0
    inference: float = 0.0
    reconstruction: float = 0.0

    @property
    def total(self) -> float:
        return sum((
            self.localization,
            self.moments,
            self.derivatives,
            self.projection,
            self.inference,
            self.reconstruction,
        ))

    def as_dict(self) -> dict[str, float]:
        return {
            "localization": self.localization,
            "moments": self.moments,
            "derivatives": self.derivatives,
            "projection": self.projection,
            "inference": self.inference,
            "reconstruction": self.reconstruction,
            "total": self.total,
        }


@dataclass(slots=True)
class RegularizationInfo:
    method: str = "ridge"
    omega_gg: float = 0.0
    residual_covariance: float = 0.0
    information: float = 0.0


@dataclass(slots=True)
class GMMResult:
    method: str
    theta: Array
    preliminary_theta: Array
    covariance: Array
    standard_errors: Array
    success: bool
    message: str
    objective_value: float | None = None
    update: Array | None = None
    raw_update: Array | None = None
    full_score_update: Array | None = None
    residual_only_update: Array | None = None
    tractable_score: Array | None = None
    residual_score: Array | None = None
    tractable_weight: Array | None = None
    tractable_foc_norm: float | None = None
    update_difference_norm: float | None = None
    damping_factor: float = 1.0
    initial_tractable_theta: Array | None = None
    gbar: Array | None = None
    hbar: Array | None = None
    nubar: Array | None = None
    omega_gg: Array | None = None
    omega_hg: Array | None = None
    residual_covariance: Array | None = None
    projection: Array | None = None
    G: Array | None = None
    H: Array | None = None
    R: Array | None = None
    information: Array | None = None
    orthogonality_residual: Array | None = None
    condition_numbers: dict[str, float] = field(default_factory=dict)
    effective_ranks: dict[str, int] = field(default_factory=dict)
    counts: EvaluationCounts = field(default_factory=EvaluationCounts)
    timings: StageTimings = field(default_factory=StageTimings)
    regularization: RegularizationInfo = field(default_factory=RegularizationInfo)
    warnings: list[str] = field(default_factory=list)
    optimizer_result: Any = None
    reconstruction: Any = None

    def summary(self) -> str:
        lines = [
            f"Method: {self.method}",
            f"Success: {self.success}",
            f"Message: {self.message}",
            f"Estimate: {np.array2string(self.theta, precision=6)}",
            f"Std. errors: {np.array2string(self.standard_errors, precision=6)}",
        ]
        if self.update is not None:
            lines.append(f"Update norm: {np.linalg.norm(self.update):.6g}")
        if self.tractable_foc_norm is not None:
            lines.append(
                "Tractable FOC norm: "
                f"{self.tractable_foc_norm:.6g}"
            )
        if self.update_difference_norm is not None:
            lines.append(
                "Full/residual update difference: "
                f"{self.update_difference_norm:.6g}"
            )
        if self.condition_numbers:
            lines.append(f"Condition numbers: {self.condition_numbers}")
        lines.append(f"Evaluation counts: {self.counts.as_dict()}")
        lines.append(f"Timings: {self.timings.as_dict()}")
        if self.warnings:
            lines.append("Warnings: " + "; ".join(self.warnings))
        return "\n".join(lines)

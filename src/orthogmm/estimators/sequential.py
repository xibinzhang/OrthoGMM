r"""Sequential projected corrections for local GMM systems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..linalg import solve
from ..operators.projected_information import ProjectedInformationResult


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SequentialProjectionResult:
    """Result of one damped sequential projected correction."""

    theta_initial: FloatArray
    theta_updated: FloatArray
    raw_direction: FloatArray
    applied_step: FloatArray
    tractable_score: FloatArray
    demanding_score: FloatArray
    total_score: FloatArray
    damping: float
    radius: float | None
    radius_clipped: bool

    def __post_init__(self) -> None:
        names = (
            "theta_initial",
            "theta_updated",
            "raw_direction",
            "applied_step",
            "tractable_score",
            "demanding_score",
            "total_score",
        )
        arrays = {
            name: np.asarray(getattr(self, name), dtype=float).reshape(-1)
            for name in names
        }

        dimension = arrays["theta_initial"].size
        if dimension < 1:
            raise ValueError("The parameter vector cannot be empty.")
        if any(array.size != dimension for array in arrays.values()):
            raise ValueError(
                "All parameter and score vectors must have the same dimension."
            )
        if not all(np.all(np.isfinite(array)) for array in arrays.values()):
            raise ValueError(
                "SequentialProjectionResult contains non-finite values."
            )

        for name, array in arrays.items():
            object.__setattr__(self, name, array)

    @property
    def raw_direction_norm(self) -> float:
        return float(np.linalg.norm(self.raw_direction))

    @property
    def applied_step_norm(self) -> float:
        return float(np.linalg.norm(self.applied_step))

    @property
    def score_norm(self) -> float:
        return float(np.linalg.norm(self.total_score))


class SequentialProjectionCorrection:
    r"""Compute one Newton-type projected GMM correction.

    With Jacobians defined as derivatives of the moments, the local GMM
    first-order condition has score

    .. math::

       s(\theta)
       =
       \widetilde G^\prime \Omega_{\widetilde g\widetilde g}^{-1}
       \bar{\widetilde g}
       +
       R^\prime S^{-1}\bar\nu.

    The Newton direction is therefore :math:`-J^{-1}s(\theta)`.
    """

    def __init__(
        self,
        *,
        damping: float = 1.0,
        radius: float | None = None,
    ) -> None:
        if not 0.0 < damping <= 1.0:
            raise ValueError("damping must lie in (0, 1].")
        if radius is not None and radius <= 0.0:
            raise ValueError("radius must be positive.")

        self.damping = float(damping)
        self.radius = None if radius is None else float(radius)

    def apply(
        self,
        theta: FloatArray,
        projected: ProjectedInformationResult,
    ) -> SequentialProjectionResult:
        """Apply one damped and optionally radius-limited correction."""

        theta_array = np.asarray(theta, dtype=float).reshape(-1)
        if theta_array.size != projected.parameter_dimension:
            raise ValueError(
                "theta dimension does not match projected information."
            )
        if not np.all(np.isfinite(theta_array)):
            raise ValueError("theta contains non-finite values.")

        basis = projected.basis_result
        g_reduced = basis.reduced_moments
        G_reduced = basis.reduced_jacobian

        g_bar = g_reduced.mean(axis=0)
        nu_bar = projected.residual_moments.mean(axis=0)

        tractable_score = G_reduced.T @ solve(
            projected.omega_gg,
            g_bar,
        )
        demanding_score = projected.residual_jacobian.T @ solve(
            projected.schur_complement,
            nu_bar,
        )
        total_score = tractable_score + demanding_score

        raw_direction = -solve(
            projected.information,
            total_score,
        )
        step = self.damping * raw_direction

        radius_clipped = False
        if self.radius is not None:
            norm = float(np.linalg.norm(step))
            if norm > self.radius:
                step = step * (self.radius / norm)
                radius_clipped = True

        theta_updated = theta_array + step

        return SequentialProjectionResult(
            theta_initial=theta_array,
            theta_updated=theta_updated,
            raw_direction=raw_direction,
            applied_step=step,
            tractable_score=tractable_score,
            demanding_score=demanding_score,
            total_score=total_score,
            damping=self.damping,
            radius=self.radius,
            radius_clipped=radius_clipped,
        )

    __call__ = apply

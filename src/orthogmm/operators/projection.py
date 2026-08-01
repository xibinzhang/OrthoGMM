from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..covariance import covariance_blocks_iid, residual_covariance_iid
from ..linalg import solve, stable_matrix
from ..types import Array


@dataclass(slots=True)
class ProjectionResult:
    """Estimated objects in the finite-dimensional orthogonal projection."""

    coefficient: Array
    residuals: Array
    residual_covariance: Array
    residualized_jacobian: Array
    information: Array
    projected_score: Array
    orthogonality_residual: Array
    omega_gg: Array
    omega_hg: Array
    condition_numbers: dict[str, float]
    effective_ranks: dict[str, int]
    ridge_levels: dict[str, float]


class OrthogonalProjection:
    """Construct the projected efficient GMM objects ``(B, S, R, J)``.

    This object implements the population/sample algebra in Section 3 of the
    paper. It deliberately contains no optimizer and no model-specific code.
    """

    def __init__(self, *, ridge: float = 0.0, condition_limit: float = 1e12):
        self.ridge = float(ridge)
        self.condition_limit = float(condition_limit)
        self.result_: ProjectionResult | None = None

    def fit(self, g: Array, h: Array, G: Array, H: Array) -> ProjectionResult:
        g = np.asarray(g, dtype=float)
        h = np.asarray(h, dtype=float)
        G = np.asarray(G, dtype=float)
        H = np.asarray(H, dtype=float)
        if g.ndim != 2 or h.ndim != 2:
            raise ValueError("g and h must be unit-by-moment matrices.")
        if g.shape[0] != h.shape[0]:
            raise ValueError("g and h must use the same statistical units.")
        if G.shape[1] != H.shape[1]:
            raise ValueError("G and H must have the same parameter dimension.")
        if G.shape[0] != g.shape[1] or H.shape[0] != h.shape[1]:
            raise ValueError("Moment and Jacobian dimensions are inconsistent.")

        omega_gg_raw, omega_hg, _ = covariance_blocks_iid(g, h)
        omega_gg = stable_matrix(
            omega_gg_raw, ridge=self.ridge, condition_limit=self.condition_limit
        )
        B = solve(omega_gg.value.T, omega_hg.T).T
        nu = h - g @ B.T
        S = stable_matrix(
            residual_covariance_iid(nu),
            ridge=self.ridge,
            condition_limit=self.condition_limit,
        )
        R = H - B @ G
        J = stable_matrix(
            G.T @ solve(omega_gg.value, G) + R.T @ solve(S.value, R),
            ridge=self.ridge,
            condition_limit=self.condition_limit,
        )

        gbar = g.mean(axis=0)
        nubar = nu.mean(axis=0)
        score = G.T @ solve(omega_gg.value, gbar) + R.T @ solve(S.value, nubar)
        gc = g - gbar
        nuc = nu - nubar
        orthogonality = gc.T @ nuc / g.shape[0]

        result = ProjectionResult(
            coefficient=B,
            residuals=nu,
            residual_covariance=S.value,
            residualized_jacobian=R,
            information=J.value,
            projected_score=score,
            orthogonality_residual=orthogonality,
            omega_gg=omega_gg.value,
            omega_hg=omega_hg,
            condition_numbers={
                "omega_gg": omega_gg.condition_number,
                "residual_covariance": S.condition_number,
                "information": J.condition_number,
            },
            effective_ranks={
                "omega_gg": omega_gg.effective_rank,
                "residual_covariance": S.effective_rank,
                "information": J.effective_rank,
            },
            ridge_levels={
                "omega_gg": omega_gg.ridge,
                "residual_covariance": S.ridge,
                "information": J.ridge,
            },
        )
        self.result_ = result
        return result

    @property
    def coefficient_(self) -> Array:
        if self.result_ is None:
            raise RuntimeError("Projection has not been fitted.")
        return self.result_.coefficient

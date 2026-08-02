r"""Rank-sensitivity diagnostics for projected information systems."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from ..linalg import solve, stable_matrix
from ..operators.basis import TractableMomentBasis


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RankAuditRow:
    """Diagnostics for one retained tractable-moment rank."""

    rank: int
    empirical_rank: int
    explained_variance_ratio: float
    discarded_energy_ratio: float
    condition_omega_gg: float
    condition_schur: float
    condition_information: float
    raw_rank_omega_gg: int
    raw_rank_schur: int
    raw_rank_information: int
    stabilized_rank_omega_gg: int
    stabilized_rank_schur: int
    stabilized_rank_information: int
    ridge_omega_gg: float
    ridge_schur: float
    ridge_information: float
    orthogonality_norm: float
    projection_norm: float
    residual_jacobian_norm: float
    elapsed_seconds: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _numerical_rank(matrix: FloatArray) -> int:
    """Return NumPy's scale-aware numerical matrix rank."""

    return int(np.linalg.matrix_rank(np.asarray(matrix, dtype=float)))


class RankAudit:
    """Evaluate projected-information diagnostics over retained ranks."""

    def __init__(
        self,
        *,
        minimum_rank: int = 1,
        maximum_rank: int | None = None,
        center: bool = True,
        ridge: float = 0.0,
        singular_value_tolerance: float | None = None,
    ) -> None:
        if minimum_rank < 1:
            raise ValueError("minimum_rank must be positive.")
        if maximum_rank is not None and maximum_rank < minimum_rank:
            raise ValueError(
                "maximum_rank must be at least minimum_rank."
            )
        if ridge < 0:
            raise ValueError("ridge must be nonnegative.")

        self.minimum_rank = int(minimum_rank)
        self.maximum_rank = maximum_rank
        self.center = bool(center)
        self.ridge = float(ridge)
        self.singular_value_tolerance = singular_value_tolerance

    @staticmethod
    def _validate_inputs(
        g: FloatArray,
        h: FloatArray,
        G: FloatArray,
        H: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        g_array = np.asarray(g, dtype=float)
        h_array = np.asarray(h, dtype=float)
        G_array = np.asarray(G, dtype=float)
        H_array = np.asarray(H, dtype=float)

        if g_array.ndim != 2 or h_array.ndim != 2:
            raise ValueError("g and h must be two-dimensional.")
        if g_array.shape[0] != h_array.shape[0]:
            raise ValueError(
                "g and h must share the same statistical units."
            )
        if G_array.ndim != 2 or H_array.ndim != 2:
            raise ValueError("G and H must be two-dimensional.")
        if G_array.shape[0] != g_array.shape[1]:
            raise ValueError(
                "G must have one row per tractable moment."
            )
        if H_array.shape[0] != h_array.shape[1]:
            raise ValueError(
                "H must have one row per demanding moment."
            )
        if G_array.shape[1] != H_array.shape[1]:
            raise ValueError(
                "G and H must have the same parameter dimension."
            )

        arrays = (g_array, h_array, G_array, H_array)
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("Inputs contain non-finite values.")

        return g_array, h_array, G_array, H_array

    def run(
        self,
        g: FloatArray,
        h: FloatArray,
        G: FloatArray,
        H: FloatArray,
    ) -> list[RankAuditRow]:
        """Compute diagnostics for all requested retained ranks."""

        g_array, h_array, G_array, H_array = self._validate_inputs(
            g,
            h,
            G,
            H,
        )

        full_basis = TractableMomentBasis(
            singular_value_tolerance=self.singular_value_tolerance,
            center=self.center,
        ).fit(g_array, G_array)

        empirical_rank = full_basis.empirical_rank
        maximum_rank = (
            empirical_rank
            if self.maximum_rank is None
            else min(self.maximum_rank, empirical_rank)
        )

        if self.minimum_rank > maximum_rank:
            raise ValueError(
                "minimum_rank exceeds the empirical tractable rank."
            )

        rows: list[RankAuditRow] = []
        n = g_array.shape[0]

        for rank in range(self.minimum_rank, maximum_rank + 1):
            start = perf_counter()

            basis = TractableMomentBasis(
                rank=rank,
                singular_value_tolerance=self.singular_value_tolerance,
                center=self.center,
            ).fit(g_array, G_array)

            g_reduced = basis.reduced_moments
            G_reduced = basis.reduced_jacobian

            if self.center:
                gc = g_reduced - g_reduced.mean(axis=0)
                hc = h_array - h_array.mean(axis=0)
            else:
                gc = g_reduced
                hc = h_array

            omega_gg_raw = gc.T @ gc / n
            omega_hg = hc.T @ gc / n
            omega_hh_raw = hc.T @ hc / n

            omega_gg_stable = stable_matrix(
                omega_gg_raw,
                ridge=self.ridge,
            )
            B_reduced = solve(
                omega_gg_stable.value,
                omega_hg.T,
            ).T

            residual_moments = h_array - g_reduced @ B_reduced.T
            residual_centered = (
                residual_moments - residual_moments.mean(axis=0)
                if self.center
                else residual_moments
            )

            S_raw = omega_hh_raw - B_reduced @ omega_hg.T
            S_stable = stable_matrix(
                S_raw,
                ridge=self.ridge,
            )

            R = H_array - B_reduced @ G_reduced

            first_information = G_reduced.T @ solve(
                omega_gg_stable.value,
                G_reduced,
            )
            second_information = R.T @ solve(
                S_stable.value,
                R,
            )
            J_raw = first_information + second_information
            J_stable = stable_matrix(
                J_raw,
                ridge=self.ridge,
            )

            orthogonality = float(
                np.linalg.norm(
                    residual_centered.T @ gc / n
                )
            )

            rows.append(
                RankAuditRow(
                    rank=rank,
                    empirical_rank=empirical_rank,
                    explained_variance_ratio=(
                        basis.explained_variance_ratio
                    ),
                    discarded_energy_ratio=(
                        basis.discarded_energy_ratio
                    ),
                    condition_omega_gg=(
                        omega_gg_stable.condition_number
                    ),
                    condition_schur=(
                        S_stable.condition_number
                    ),
                    condition_information=(
                        J_stable.condition_number
                    ),
                    raw_rank_omega_gg=_numerical_rank(
                        omega_gg_raw
                    ),
                    raw_rank_schur=_numerical_rank(S_raw),
                    raw_rank_information=_numerical_rank(J_raw),
                    stabilized_rank_omega_gg=(
                        omega_gg_stable.effective_rank
                    ),
                    stabilized_rank_schur=(
                        S_stable.effective_rank
                    ),
                    stabilized_rank_information=(
                        J_stable.effective_rank
                    ),
                    ridge_omega_gg=omega_gg_stable.ridge,
                    ridge_schur=S_stable.ridge,
                    ridge_information=J_stable.ridge,
                    orthogonality_norm=orthogonality,
                    projection_norm=float(
                        np.linalg.norm(B_reduced)
                    ),
                    residual_jacobian_norm=float(
                        np.linalg.norm(R)
                    ),
                    elapsed_seconds=perf_counter() - start,
                )
            )

        return rows

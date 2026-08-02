r"""Projected information objects for sequential orthogonal projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..linalg import solve, stable_matrix
from .basis import TractableMomentBasis, TractableMomentBasisResult


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ProjectedInformationResult:
    """Estimated projection and efficient-information objects."""

    projection: FloatArray
    reduced_projection: FloatArray
    residual_moments: FloatArray
    omega_gg: FloatArray
    omega_hg: FloatArray
    schur_complement: FloatArray
    residual_jacobian: FloatArray
    information: FloatArray
    basis_result: TractableMomentBasisResult
    condition_numbers: dict[str, float]
    effective_ranks: dict[str, int]
    regularization: dict[str, float]

    def __post_init__(self) -> None:
        B = np.asarray(self.projection, dtype=float)
        B_reduced = np.asarray(
            self.reduced_projection,
            dtype=float,
        )
        nu = np.asarray(self.residual_moments, dtype=float)
        omega_gg = np.asarray(self.omega_gg, dtype=float)
        omega_hg = np.asarray(self.omega_hg, dtype=float)
        S = np.asarray(self.schur_complement, dtype=float)
        R = np.asarray(self.residual_jacobian, dtype=float)
        J = np.asarray(self.information, dtype=float)

        arrays = (
            B,
            B_reduced,
            nu,
            omega_gg,
            omega_hg,
            S,
            R,
            J,
        )
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError(
                "ProjectedInformationResult contains non-finite values."
            )

        qh = S.shape[0]
        qg_original = self.basis_result.original_dimension
        qg_reduced = self.basis_result.retained_rank

        if B.shape != (qh, qg_original):
            raise ValueError(
                "projection has an incompatible shape."
            )
        if B_reduced.shape != (qh, qg_reduced):
            raise ValueError(
                "reduced_projection has an incompatible shape."
            )
        if omega_gg.shape != (qg_reduced, qg_reduced):
            raise ValueError("omega_gg has an incompatible shape.")
        if omega_hg.shape != (qh, qg_reduced):
            raise ValueError("omega_hg has an incompatible shape.")
        if nu.ndim != 2 or nu.shape[1] != qh:
            raise ValueError(
                "residual_moments has an incompatible shape."
            )
        if R.shape[0] != qh:
            raise ValueError(
                "residual_jacobian has an incompatible shape."
            )
        if J.ndim != 2 or J.shape[0] != J.shape[1]:
            raise ValueError("information must be square.")
        if J.shape[0] != R.shape[1]:
            raise ValueError(
                "information dimension must match parameter dimension."
            )

        object.__setattr__(self, "projection", B)
        object.__setattr__(
            self,
            "reduced_projection",
            B_reduced,
        )
        object.__setattr__(self, "residual_moments", nu)
        object.__setattr__(self, "omega_gg", omega_gg)
        object.__setattr__(self, "omega_hg", omega_hg)
        object.__setattr__(self, "schur_complement", S)
        object.__setattr__(self, "residual_jacobian", R)
        object.__setattr__(self, "information", J)

    @property
    def n_units(self) -> int:
        return int(self.residual_moments.shape[0])

    @property
    def original_tractable_dimension(self) -> int:
        return self.basis_result.original_dimension

    @property
    def retained_tractable_rank(self) -> int:
        return self.basis_result.retained_rank

    @property
    def n_demanding_moments(self) -> int:
        return int(self.schur_complement.shape[0])

    @property
    def parameter_dimension(self) -> int:
        return int(self.information.shape[0])


class ProjectedInformationOperator:
    """Estimate projected information on a reduced tractable moment basis."""

    def __init__(
        self,
        *,
        rank: int | None = None,
        singular_value_tolerance: float | None = None,
        explained_variance: float | None = None,
        center: bool = True,
        ridge: float = 0.0,
    ) -> None:
        if ridge < 0:
            raise ValueError("ridge must be nonnegative.")

        self.basis_operator = TractableMomentBasis(
            rank=rank,
            singular_value_tolerance=singular_value_tolerance,
            explained_variance=explained_variance,
            center=center,
        )
        self.center = bool(center)
        self.ridge = float(ridge)

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

    def fit(
        self,
        g: FloatArray,
        h: FloatArray,
        G: FloatArray,
        H: FloatArray,
    ) -> ProjectedInformationResult:
        """Estimate ``B``, ``S``, ``R``, and ``J`` on the retained basis."""

        g_array, h_array, G_array, H_array = self._validate_inputs(
            g,
            h,
            G,
            H,
        )

        basis_result = self.basis_operator.fit(
            g_array,
            G_array,
        )
        g_reduced = basis_result.reduced_moments
        G_reduced = basis_result.reduced_jacobian

        if self.center:
            gc = g_reduced - g_reduced.mean(axis=0)
            hc = h_array - h_array.mean(axis=0)
        else:
            gc = g_reduced
            hc = h_array

        n = g_array.shape[0]
        omega_gg_raw = (gc.T @ gc) / n
        omega_hg = (hc.T @ gc) / n
        omega_hh_raw = (hc.T @ hc) / n

        omega_gg_stable = stable_matrix(
            omega_gg_raw,
            ridge=self.ridge,
        )

        B_reduced = solve(
            omega_gg_stable.value,
            omega_hg.T,
        ).T
        B = B_reduced @ basis_result.basis.T

        residual_moments = h_array - g_reduced @ B_reduced.T

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

        return ProjectedInformationResult(
            projection=B,
            reduced_projection=B_reduced,
            residual_moments=residual_moments,
            omega_gg=omega_gg_stable.value,
            omega_hg=omega_hg,
            schur_complement=S_stable.value,
            residual_jacobian=R,
            information=J_stable.value,
            basis_result=basis_result,
            condition_numbers={
                "omega_gg": omega_gg_stable.condition_number,
                "schur_complement": S_stable.condition_number,
                "information": J_stable.condition_number,
            },
            effective_ranks={
                "omega_gg": omega_gg_stable.effective_rank,
                "schur_complement": S_stable.effective_rank,
                "information": J_stable.effective_rank,
            },
            regularization={
                "omega_gg": omega_gg_stable.ridge,
                "schur_complement": S_stable.ridge,
                "information": J_stable.ridge,
            },
        )

    __call__ = fit

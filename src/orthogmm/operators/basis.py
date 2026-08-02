r"""Rank-revealing bases for tractable moment systems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TractableMomentBasisResult:
    """SVD representation of the empirical tractable moment space."""

    reduced_moments: FloatArray
    reduced_jacobian: FloatArray
    basis: FloatArray
    singular_values: FloatArray
    empirical_rank: int
    retained_rank: int
    explained_variance_ratio: float
    discarded_energy_ratio: float
    centered: bool
    mean: FloatArray

    def __post_init__(self) -> None:
        reduced_moments = np.asarray(self.reduced_moments, dtype=float)
        reduced_jacobian = np.asarray(self.reduced_jacobian, dtype=float)
        basis = np.asarray(self.basis, dtype=float)
        singular_values = np.asarray(
            self.singular_values,
            dtype=float,
        ).reshape(-1)
        mean = np.asarray(self.mean, dtype=float).reshape(-1)

        if reduced_moments.ndim != 2:
            raise ValueError("reduced_moments must be two-dimensional.")
        if reduced_jacobian.ndim != 2:
            raise ValueError("reduced_jacobian must be two-dimensional.")
        if basis.ndim != 2:
            raise ValueError("basis must be two-dimensional.")
        if basis.shape[1] != self.retained_rank:
            raise ValueError(
                "basis column count must equal retained_rank."
            )
        if reduced_moments.shape[1] != self.retained_rank:
            raise ValueError(
                "reduced_moments column count must equal retained_rank."
            )
        if reduced_jacobian.shape[0] != self.retained_rank:
            raise ValueError(
                "reduced_jacobian row count must equal retained_rank."
            )
        if basis.shape[0] != mean.size:
            raise ValueError(
                "basis row count must equal the original moment dimension."
            )
        if self.retained_rank > self.empirical_rank:
            raise ValueError(
                "retained_rank cannot exceed empirical_rank."
            )
        if not 0.0 <= self.explained_variance_ratio <= 1.0 + 1e-12:
            raise ValueError(
                "explained_variance_ratio must lie in [0, 1]."
            )
        if not 0.0 <= self.discarded_energy_ratio <= 1.0 + 1e-12:
            raise ValueError(
                "discarded_energy_ratio must lie in [0, 1]."
            )

        arrays = (
            reduced_moments,
            reduced_jacobian,
            basis,
            singular_values,
            mean,
        )
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError(
                "TractableMomentBasisResult contains non-finite values."
            )

        object.__setattr__(
            self,
            "reduced_moments",
            reduced_moments,
        )
        object.__setattr__(
            self,
            "reduced_jacobian",
            reduced_jacobian,
        )
        object.__setattr__(self, "basis", basis)
        object.__setattr__(
            self,
            "singular_values",
            singular_values,
        )
        object.__setattr__(self, "mean", mean)

    @property
    def original_dimension(self) -> int:
        return int(self.basis.shape[0])

    @property
    def n_units(self) -> int:
        return int(self.reduced_moments.shape[0])


class TractableMomentBasis:
    """Construct a reduced orthonormal basis for tractable moments."""

    def __init__(
        self,
        *,
        rank: int | None = None,
        singular_value_tolerance: float | None = None,
        explained_variance: float | None = None,
        center: bool = True,
    ) -> None:
        if rank is not None and rank < 1:
            raise ValueError("rank must be positive.")
        if (
            singular_value_tolerance is not None
            and singular_value_tolerance < 0
        ):
            raise ValueError(
                "singular_value_tolerance must be nonnegative."
            )
        if explained_variance is not None and not (
            0.0 < explained_variance <= 1.0
        ):
            raise ValueError(
                "explained_variance must lie in (0, 1]."
            )
        if rank is not None and explained_variance is not None:
            raise ValueError(
                "Specify either rank or explained_variance, not both."
            )

        self.rank = rank
        self.singular_value_tolerance = singular_value_tolerance
        self.explained_variance = explained_variance
        self.center = bool(center)

    def fit(
        self,
        moments: FloatArray,
        jacobian: FloatArray,
    ) -> TractableMomentBasisResult:
        """Reduce a tractable moment system using the right singular basis."""

        g = np.asarray(moments, dtype=float)
        G = np.asarray(jacobian, dtype=float)

        if g.ndim != 2:
            raise ValueError("moments must be two-dimensional.")
        if G.ndim != 2:
            raise ValueError("jacobian must be two-dimensional.")
        if G.shape[0] != g.shape[1]:
            raise ValueError(
                "jacobian must have one row per original moment."
            )
        if g.shape[0] < 2:
            raise ValueError(
                "At least two statistical units are required."
            )
        if not np.all(np.isfinite(g)) or not np.all(np.isfinite(G)):
            raise ValueError("Inputs contain non-finite values.")

        mean = g.mean(axis=0) if self.center else np.zeros(g.shape[1])
        gc = g - mean if self.center else g.copy()

        _, singular_values, vt = np.linalg.svd(
            gc,
            full_matrices=False,
        )

        if singular_values.size == 0:
            raise ValueError("No singular values were computed.")

        default_tol = (
            max(gc.shape)
            * np.finfo(float).eps
            * singular_values[0]
        )
        tolerance = (
            default_tol
            if self.singular_value_tolerance is None
            else self.singular_value_tolerance
        )
        empirical_rank = int(
            np.sum(singular_values > tolerance)
        )

        if empirical_rank < 1:
            raise ValueError(
                "The tractable moment matrix has empirical rank zero."
            )

        squared = singular_values[:empirical_rank] ** 2
        total_energy = float(squared.sum())

        if self.rank is not None:
            retained_rank = min(self.rank, empirical_rank)
        elif self.explained_variance is not None:
            cumulative = np.cumsum(squared) / total_energy
            retained_rank = int(
                np.searchsorted(
                    cumulative,
                    self.explained_variance,
                    side="left",
                )
                + 1
            )
        else:
            retained_rank = empirical_rank

        basis = vt[:retained_rank].T
        reduced_moments = g @ basis
        reduced_jacobian = basis.T @ G

        retained_energy = float(
            np.sum(singular_values[:retained_rank] ** 2)
        )
        explained_ratio = retained_energy / total_energy
        discarded_ratio = 1.0 - explained_ratio

        return TractableMomentBasisResult(
            reduced_moments=reduced_moments,
            reduced_jacobian=reduced_jacobian,
            basis=basis,
            singular_values=singular_values,
            empirical_rank=empirical_rank,
            retained_rank=retained_rank,
            explained_variance_ratio=explained_ratio,
            discarded_energy_ratio=discarded_ratio,
            centered=self.center,
            mean=mean,
        )

    __call__ = fit

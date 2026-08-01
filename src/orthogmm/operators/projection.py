from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..covariance import (
    covariance_blocks_cluster,
    covariance_blocks_iid,
    residual_covariance_cluster,
    residual_covariance_iid,
)
from ..linalg import solve, stable_matrix
from ..types import Array


CovarianceType = Literal["iid", "cluster"]


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

    This object implements the sample counterparts of the projection algebra
    developed in Section 3 of the paper. It contains no optimizer and no
    model-specific code.
    """

    def __init__(
        self,
        *,
        ridge: float = 0.0,
        condition_limit: float = 1e12,
    ) -> None:
        self.ridge = float(ridge)
        self.condition_limit = float(condition_limit)
        self.result_: ProjectionResult | None = None

    def fit(
        self,
        g: Array,
        h: Array,
        G: Array,
        H: Array,
        *,
        covariance_type: CovarianceType = "iid",
        clusters: Array | None = None,
    ) -> ProjectionResult:
        """Estimate the orthogonal projection and projected GMM objects.

        Parameters
        ----------
        g
            Unit-by-tractable-moment matrix.
        h
            Unit-by-demanding-moment matrix.
        G
            Jacobian of the tractable moments.
        H
            Jacobian of the demanding moments.
        covariance_type
            Either ``"iid"`` or ``"cluster"``.
        clusters
            Cluster identifiers. Required when ``covariance_type="cluster"``.
        """
        g = np.asarray(g, dtype=float)
        h = np.asarray(h, dtype=float)
        G = np.asarray(G, dtype=float)
        H = np.asarray(H, dtype=float)

        self._validate_inputs(
            g=g,
            h=h,
            G=G,
            H=H,
            covariance_type=covariance_type,
            clusters=clusters,
        )

        cluster_ids: Array | None

        if covariance_type == "iid":
            omega_gg_raw, omega_hg, _ = covariance_blocks_iid(g, h)
            cluster_ids = None
        else:
            cluster_ids = np.asarray(clusters)
            omega_gg_raw, omega_hg, _ = covariance_blocks_cluster(
                g,
                h,
                cluster_ids,
            )

        omega_gg = stable_matrix(
            omega_gg_raw,
            ridge=self.ridge,
            condition_limit=self.condition_limit,
        )

        # B = Omega_hg Omega_gg^{-1}
        coefficient = solve(
            omega_gg.value.T,
            omega_hg.T,
        ).T

        # nu_i = h_i - B g_i
        residuals = h - g @ coefficient.T

        if covariance_type == "iid":
            residual_covariance_raw = residual_covariance_iid(residuals)
        else:
            assert cluster_ids is not None
            residual_covariance_raw = residual_covariance_cluster(
                residuals,
                cluster_ids,
            )

        residual_covariance = stable_matrix(
            residual_covariance_raw,
            ridge=self.ridge,
            condition_limit=self.condition_limit,
        )

        # R = H - B G
        residualized_jacobian = H - coefficient @ G

        # J = G' Omega_gg^{-1} G + R' S^{-1} R
        information_raw = (
            G.T @ solve(omega_gg.value, G)
            + residualized_jacobian.T
            @ solve(
                residual_covariance.value,
                residualized_jacobian,
            )
        )

        information = stable_matrix(
            information_raw,
            ridge=self.ridge,
            condition_limit=self.condition_limit,
        )

        gbar = g.mean(axis=0)
        residual_mean = residuals.mean(axis=0)

        # psi = G' Omega_gg^{-1} gbar + R' S^{-1} nubar
        projected_score = (
            G.T @ solve(omega_gg.value, gbar)
            + residualized_jacobian.T
            @ solve(
                residual_covariance.value,
                residual_mean,
            )
        )

        # Sample orthogonality diagnostic.
        centred_g = g - gbar
        centred_residuals = residuals - residual_mean
        orthogonality_residual = (
            centred_g.T @ centred_residuals / g.shape[0]
        )

        result = ProjectionResult(
            coefficient=coefficient,
            residuals=residuals,
            residual_covariance=residual_covariance.value,
            residualized_jacobian=residualized_jacobian,
            information=information.value,
            projected_score=projected_score,
            orthogonality_residual=orthogonality_residual,
            omega_gg=omega_gg.value,
            omega_hg=omega_hg,
            condition_numbers={
                "omega_gg": omega_gg.condition_number,
                "residual_covariance": (
                    residual_covariance.condition_number
                ),
                "information": information.condition_number,
            },
            effective_ranks={
                "omega_gg": omega_gg.effective_rank,
                "residual_covariance": (
                    residual_covariance.effective_rank
                ),
                "information": information.effective_rank,
            },
            ridge_levels={
                "omega_gg": omega_gg.ridge,
                "residual_covariance": residual_covariance.ridge,
                "information": information.ridge,
            },
        )

        self.result_ = result
        return result

    @staticmethod
    def _validate_inputs(
        *,
        g: Array,
        h: Array,
        G: Array,
        H: Array,
        covariance_type: CovarianceType,
        clusters: Array | None,
    ) -> None:
        if g.ndim != 2 or h.ndim != 2:
            raise ValueError(
                "g and h must be unit-by-moment matrices."
            )

        if g.shape[0] != h.shape[0]:
            raise ValueError(
                "g and h must use the same statistical units."
            )

        if G.ndim != 2 or H.ndim != 2:
            raise ValueError("G and H must be two-dimensional matrices.")

        if G.shape[1] != H.shape[1]:
            raise ValueError(
                "G and H must have the same parameter dimension."
            )

        if G.shape[0] != g.shape[1]:
            raise ValueError(
                "The number of rows of G must equal the number "
                "of tractable moments."
            )

        if H.shape[0] != h.shape[1]:
            raise ValueError(
                "The number of rows of H must equal the number "
                "of demanding moments."
            )

        if covariance_type not in ("iid", "cluster"):
            raise ValueError(
                "covariance_type must be 'iid' or 'cluster'."
            )

        if covariance_type == "cluster":
            if clusters is None:
                raise ValueError(
                    "clusters must be supplied when "
                    "covariance_type='cluster'."
                )

            cluster_array = np.asarray(clusters)

            if cluster_array.ndim != 1:
                raise ValueError(
                    "clusters must be a one-dimensional array."
                )

            if cluster_array.shape[0] != g.shape[0]:
                raise ValueError(
                    "clusters must contain one identifier per "
                    "statistical unit."
                )

    @property
    def coefficient_(self) -> Array:
        if self.result_ is None:
            raise RuntimeError("Projection has not been fitted.")
        return self.result_.coefficient

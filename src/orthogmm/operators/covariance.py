from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..covariance import covariance_blocks_cluster, covariance_blocks_iid
from ..exceptions import ModelContractError
from ..linalg import solve, stable_matrix
from ..types import Array
from .base import BaseOperator


CovarianceType = Literal["iid", "cluster"]


@dataclass(frozen=True, slots=True)
class CovarianceResult:
    """Estimated covariance operator and numerical diagnostics."""

    covariance: Array
    weight: Array
    condition_number: float
    effective_rank: int
    ridge: float
    covariance_type: CovarianceType
    n_units: int
    n_moments: int
    n_clusters: int | None = None


class CovarianceOperator(BaseOperator):
    """Estimate and stabilize a covariance matrix of unit-level moments."""

    def __init__(
        self,
        *,
        ridge: float = 0.0,
        condition_limit: float = 1e12,
    ) -> None:
        if ridge < 0:
            raise ValueError("ridge must be non-negative.")
        if condition_limit <= 1:
            raise ValueError("condition_limit must exceed 1.")
        self.ridge = float(ridge)
        self.condition_limit = float(condition_limit)
        self.result_: CovarianceResult | None = None

    def fit(
        self,
        moments: Array,
        *,
        covariance_type: CovarianceType = "iid",
        clusters: Array | None = None,
    ) -> CovarianceResult:
        moments = self._validate_moments(moments)

        if covariance_type == "iid":
            covariance_raw, _, _ = covariance_blocks_iid(moments, moments)
            n_clusters = None
        elif covariance_type == "cluster":
            cluster_ids = self._validate_clusters(
                clusters,
                n_units=moments.shape[0],
            )
            covariance_raw, _, _ = covariance_blocks_cluster(
                moments,
                moments,
                cluster_ids,
            )
            n_clusters = int(np.unique(cluster_ids).size)
        else:
            raise ValueError(
                "covariance_type must be 'iid' or 'cluster'."
            )

        stabilized = stable_matrix(
            covariance_raw,
            ridge=self.ridge,
            condition_limit=self.condition_limit,
        )
        weight = solve(
            stabilized.value,
            np.eye(stabilized.value.shape[0]),
        )

        result = CovarianceResult(
            covariance=stabilized.value,
            weight=weight,
            condition_number=stabilized.condition_number,
            effective_rank=stabilized.effective_rank,
            ridge=stabilized.ridge,
            covariance_type=covariance_type,
            n_units=int(moments.shape[0]),
            n_moments=int(moments.shape[1]),
            n_clusters=n_clusters,
        )
        self.result_ = result
        return result

    @staticmethod
    def _validate_moments(moments: Array) -> Array:
        moments = np.asarray(moments, dtype=float)
        if moments.ndim == 1:
            moments = moments[:, None]
        if moments.ndim != 2:
            raise ModelContractError(
                "moments must be a unit-by-moment matrix."
            )
        if moments.shape[0] < 2:
            raise ModelContractError(
                "moments must contain at least two statistical units."
            )
        if moments.shape[1] < 1:
            raise ModelContractError(
                "moments must contain at least one moment condition."
            )
        if not np.all(np.isfinite(moments)):
            raise ModelContractError(
                "moments contain non-finite values."
            )
        return moments

    @staticmethod
    def _validate_clusters(
        clusters: Array | None,
        *,
        n_units: int,
    ) -> Array:
        if clusters is None:
            raise ModelContractError(
                "clusters must be supplied for cluster covariance."
            )
        cluster_ids = np.asarray(clusters)
        if cluster_ids.ndim != 1:
            raise ModelContractError(
                "clusters must be one-dimensional."
            )
        if cluster_ids.shape[0] != n_units:
            raise ModelContractError(
                "clusters must contain one identifier per statistical unit."
            )
        if np.unique(cluster_ids).size < 2:
            raise ModelContractError(
                "Cluster covariance requires at least two clusters."
            )
        return cluster_ids

    def _require_result(self) -> CovarianceResult:
        if self.result_ is None:
            raise RuntimeError(
                "CovarianceOperator has not been fitted."
            )
        return self.result_

    @property
    def covariance_(self) -> Array:
        return self._require_result().covariance

    @property
    def weight_(self) -> Array:
        return self._require_result().weight

    @property
    def condition_number_(self) -> float:
        return self._require_result().condition_number

    @property
    def effective_rank_(self) -> int:
        return self._require_result().effective_rank

    @property
    def ridge_(self) -> float:
        return self._require_result().ridge

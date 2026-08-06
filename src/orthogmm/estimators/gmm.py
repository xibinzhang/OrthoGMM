from __future__ import annotations

from typing import Any

from ..core import fit_full_gmm, fit_seip, fit_tractable_gmm
from ..types import Array, GMMResult, MomentModel
from .base import BaseEstimator


class TractableGMM(BaseEstimator):
    """Global GMM localization using only tractable moments."""

    def fit(self, model: MomentModel, theta0: Array, **kwargs: Any) -> GMMResult:
        self.result_ = fit_tractable_gmm(model, theta0, **kwargs)
        return self.result_


class FullGMM(BaseEstimator):
    """Conventional global GMM using tractable and demanding moments."""

    def fit(self, model: MomentModel, theta0: Array, **kwargs: Any) -> GMMResult:
        self.result_ = fit_full_gmm(model, theta0, **kwargs)
        return self.result_


class SOPEstimator(BaseEstimator):
    """Sequential Orthogonal Projection estimator.

    ``fit_seip`` and ``SEIPEstimator`` are retained as backward-compatible
    aliases.
    """

    def fit(self, model: MomentModel, theta0: Array, **kwargs: Any) -> GMMResult:
        self.result_ = fit_seip(model, theta0, **kwargs)
        return self.result_


SEIPEstimator = SOPEstimator

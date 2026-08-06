"""Matched-weight residual-only SOP for the Petrin BLP application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ...linalg import solve
from ...optimization import QuadraticTrustRegion
from ...operators import OrthogonalProjection
from .local_state import PetrinLocalStateBuilder
from .model import PetrinApplicationModel


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PetrinResidualOnlySOPResult:
    """Result of one matched-weight residual-only Petrin SOP correction."""

    theta_localized: FloatArray
    theta_updated: FloatArray
    tractable_weight: FloatArray
    tractable_score: FloatArray
    residual_score: FloatArray
    projected_score: FloatArray
    full_update: FloatArray
    residual_only_update: FloatArray
    update_difference: FloatArray
    projected_information: Any
    trust_region: Any
    moment_representation_error: float
    aggregate_elapsed_seconds: float
    local_state_aggregate_elapsed_seconds: float
    micro_elapsed_seconds: float

    def __post_init__(self) -> None:
        vector_names = (
            "theta_localized",
            "theta_updated",
            "tractable_score",
            "residual_score",
            "projected_score",
            "full_update",
            "residual_only_update",
            "update_difference",
        )
        for name in vector_names:
            value = np.asarray(getattr(self, name), dtype=float).reshape(-1)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains non-finite values.")
            object.__setattr__(self, name, value)

        weight = np.asarray(self.tractable_weight, dtype=float)
        if weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
            raise ValueError("tractable_weight must be square.")
        if not np.all(np.isfinite(weight)):
            raise ValueError("tractable_weight contains non-finite values.")
        object.__setattr__(self, "tractable_weight", weight)

    @property
    def applied_step(self) -> FloatArray:
        return np.asarray(self.trust_region.step, dtype=float).reshape(-1)

    @property
    def tractable_score_norm(self) -> float:
        return float(np.linalg.norm(self.tractable_score))

    @property
    def residual_score_norm(self) -> float:
        return float(np.linalg.norm(self.residual_score))

    @property
    def full_update_norm(self) -> float:
        return float(np.linalg.norm(self.full_update))

    @property
    def residual_only_update_norm(self) -> float:
        return float(np.linalg.norm(self.residual_only_update))

    @property
    def update_difference_norm(self) -> float:
        return float(np.linalg.norm(self.update_difference))

    @property
    def relative_update_difference(self) -> float:
        denominator = max(self.residual_only_update_norm, np.finfo(float).tiny)
        return self.update_difference_norm / denominator

    @property
    def applied_step_norm(self) -> float:
        return float(np.linalg.norm(self.applied_step))


class PetrinResidualOnlySOP:
    """Apply the canonical residual-only SOP correction after Petrin localization.

    The method uses all aggregate moments. PyBLP's aggregate weighting matrix
    is passed unchanged into ``OrthogonalProjection`` so that the same fixed
    weight enters both the tractable first-order condition and projected
    information matrix. The information-metric trust region is applied to the
    residual score only. The complete transformed score is retained solely as
    a numerical diagnostic.
    """

    def __init__(
        self,
        model: PetrinApplicationModel,
        *,
        ridge: float = 1e-8,
        condition_limit: float = 1e12,
        radius: float = 1.0,
        state_builder: Any | None = None,
    ) -> None:
        if ridge < 0:
            raise ValueError("ridge must be nonnegative.")
        if condition_limit <= 1:
            raise ValueError("condition_limit must exceed one.")
        if radius <= 0:
            raise ValueError("radius must be positive.")

        self.model = model
        self.ridge = float(ridge)
        self.condition_limit = float(condition_limit)
        self.radius = float(radius)
        self.state_builder = (
            state_builder
            if state_builder is not None
            else PetrinLocalStateBuilder(model)
        )

    @staticmethod
    def _matched_weight(results: Any, qg: int) -> FloatArray:
        weight = np.asarray(results.W, dtype=float)
        expected = (qg, qg)
        if weight.shape != expected:
            raise ValueError(
                f"Expected PyBLP W with shape {expected}, got {weight.shape}."
            )
        if not np.all(np.isfinite(weight)):
            raise ValueError("PyBLP W contains non-finite values.")

        weight = 0.5 * (weight + weight.T)
        try:
            np.linalg.cholesky(weight)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "PyBLP aggregate weighting matrix must be positive definite."
            ) from exc
        return weight

    def fit(self, theta_localized: FloatArray) -> PetrinResidualOnlySOPResult:
        """Build the local demanding block once and apply residual-only SOP."""

        theta = np.asarray(theta_localized, dtype=float).reshape(-1)
        if theta.size < 1 or not np.all(np.isfinite(theta)):
            raise ValueError(
                "theta_localized must be a finite non-empty vector."
            )

        # Fixed aggregate evaluation at the already-computed localizer.
        aggregate_evaluation = self.model.evaluate_aggregate(theta)
        aggregate_results = aggregate_evaluation.pyblp.results

        # The local-state builder constructs aligned market-level aggregate and
        # micro contributions, together with analytical Jacobians.
        state = self.state_builder.build(theta)
        if state.theta.shape != theta.shape or not np.allclose(
            state.theta,
            theta,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(
                "The local state was not built at theta_localized."
            )

        g = state.tractable_moments
        qg = g.shape[1]
        weight = self._matched_weight(aggregate_results, qg)

        public_moments = np.asarray(
            aggregate_results.moments,
            dtype=float,
        ).reshape(-1)
        if public_moments.size != qg:
            raise ValueError(
                "PyBLP public moments have an incompatible dimension."
            )
        representation_error = float(
            np.linalg.norm(g.mean(axis=0) - public_moments)
        )

        projection = OrthogonalProjection(
            ridge=self.ridge,
            condition_limit=self.condition_limit,
        ).fit(
            state.tractable_moments,
            state.demanding_moments,
            state.tractable_jacobian,
            state.demanding_jacobian,
            tractable_weight=weight,
        )

        # Diagnostic transformed-score update.
        full_update = -solve(
            projection.information,
            projection.projected_score,
        )

        # Canonical residual-only SOP update.
        residual_only_update = -solve(
            projection.information,
            projection.residual_score,
        )
        update_difference = full_update - residual_only_update

        # The empirical BLP realization is regularized by an information-metric
        # trust region, applied to the residual score only.
        trust_region = QuadraticTrustRegion(
            radius=self.radius,
            metric_type="information",
            ridge=self.ridge,
        ).solve(
            projection.information,
            projection.residual_score,
            theta=theta,
        )

        theta_updated = theta + np.asarray(
            trust_region.step,
            dtype=float,
        ).reshape(-1)

        return PetrinResidualOnlySOPResult(
            theta_localized=theta,
            theta_updated=theta_updated,
            tractable_weight=weight,
            tractable_score=projection.tractable_score,
            residual_score=projection.residual_score,
            projected_score=projection.projected_score,
            full_update=full_update,
            residual_only_update=residual_only_update,
            update_difference=update_difference,
            projected_information=projection,
            trust_region=trust_region,
            moment_representation_error=representation_error,
            aggregate_elapsed_seconds=(
                aggregate_evaluation.pyblp.elapsed_seconds
            ),
            local_state_aggregate_elapsed_seconds=(
                state.aggregate_elapsed_seconds
            ),
            micro_elapsed_seconds=state.micro_elapsed_seconds,
        )

    __call__ = fit

from __future__ import annotations

from time import perf_counter
from typing import Any, Literal

import numpy as np
from scipy.optimize import minimize

from .exceptions import ModelContractError
from .jacobians import FallbackJacobian, JacobianContext
from .linalg import solve, stable_matrix
from .operators import CovarianceOperator
from .operators.projection import OrthogonalProjection
from .types import (
    Array,
    EvaluationCounts,
    GMMResult,
    MomentModel,
    RegularizationInfo,
    StageTimings,
)

CovarianceType = Literal["iid", "cluster"]


def _moments(model: MomentModel, name: str, theta: Array) -> Array:
    fn = getattr(model, name, None)
    if fn is None or not callable(fn):
        raise ModelContractError(f"Model must implement {name}(theta).")

    out = np.asarray(fn(theta), dtype=float)

    if out.ndim == 1:
        out = out[:, None]

    if out.ndim != 2:
        raise ModelContractError(
            f"{name} must return a unit-by-moment array."
        )

    if not np.all(np.isfinite(out)):
        raise ModelContractError(
            f"{name} returned non-finite values."
        )

    return out


def _mean_moments(
    model: MomentModel,
    name: str,
    theta: Array,
) -> Array:
    return _moments(model, name, theta).mean(axis=0)


def _identity_weight(q: int) -> Array:
    return np.eye(q, dtype=float)


def _validate_weight(
    weight: Array | None,
    q: int,
    name: str,
) -> Array:
    if weight is None:
        return _identity_weight(q)

    weight = np.asarray(weight, dtype=float)

    if weight.shape != (q, q):
        raise ModelContractError(
            f"{name} must have shape ({q}, {q})."
        )

    return stable_matrix(weight).value


def _jacobian(
    model: MomentModel,
    theta: Array,
    *,
    counts: EvaluationCounts,
    bounds: list[tuple[float | None, float | None]] | None,
    fd_rel_step: float | None,
    demanding: bool,
) -> Array:
    """Construct a tractable or demanding Jacobian."""

    strategy = FallbackJacobian(rel_step=fd_rel_step)

    context = JacobianContext(
        model=model,
        theta=np.asarray(theta, dtype=float),
        moment_kind="demanding" if demanding else "tractable",
        counts=counts,
        bounds=bounds,
    )

    return strategy.compute(context)


def _reconstruct(
    model: MomentModel,
    theta: Array,
    counts: EvaluationCounts,
) -> Any:
    fn = getattr(model, "reconstruct", None)

    if callable(fn):
        counts.reconstruction += 1
        return fn(theta)

    return None


def _clusters_from_model(
    model: MomentModel,
    clusters: Array | None,
) -> Array:
    if clusters is not None:
        return np.asarray(clusters)

    fn = getattr(model, "unit_ids", None)

    if callable(fn):
        try:
            return np.asarray(fn())
        except NotImplementedError:
            pass

    raise ModelContractError(
        "Cluster covariance requires clusters=... or model.unit_ids()."
    )


def fit_tractable_gmm(
    model: MomentModel,
    theta0: Array,
    *,
    weight: Array | None = None,
    bounds: list[tuple[float | None, float | None]] | None = None,
    optimizer_method: str = "L-BFGS-B",
    optimizer_options: dict[str, Any] | None = None,
    reconstruct: bool = False,
) -> GMMResult:
    theta0 = np.asarray(theta0, dtype=float)
    counts = EvaluationCounts()
    timings = StageTimings()

    probe = _moments(model, "tractable_moments", theta0)
    counts.tractable_moments += 1
    weight_matrix = _validate_weight(
        weight,
        probe.shape[1],
        "weight",
    )

    def objective(theta: Array) -> float:
        counts.tractable_objective += 1
        counts.tractable_moments += 1
        gbar = _mean_moments(
            model,
            "tractable_moments",
            theta,
        )
        return float(gbar @ weight_matrix @ gbar)

    start = perf_counter()
    optimizer_result = minimize(
        objective,
        theta0,
        method=optimizer_method,
        bounds=bounds,
        options=optimizer_options,
    )
    timings.localization = perf_counter() - start

    theta = np.asarray(optimizer_result.x, dtype=float)

    g = _moments(model, "tractable_moments", theta)
    counts.tractable_moments += 1
    gbar = g.mean(axis=0)

    start = perf_counter()

    G = _jacobian(
        model,
        theta,
        counts=counts,
        bounds=bounds,
        fd_rel_step=None,
        demanding=False,
    )

    omega_result = CovarianceOperator().fit(g)
    information = stable_matrix(
        G.T @ omega_result.weight @ G
    )
    covariance = solve(
        information.value,
        np.eye(theta.size),
    ) / g.shape[0]

    timings.inference = perf_counter() - start

    reconstruction = None
    if reconstruct:
        start = perf_counter()
        reconstruction = _reconstruct(
            model,
            theta,
            counts,
        )
        timings.reconstruction = perf_counter() - start

    warnings: list[str] = []
    if not optimizer_result.success:
        warnings.append(
            "Tractable localization did not report convergence."
        )

    return GMMResult(
        method="tractable_gmm",
        theta=theta,
        preliminary_theta=theta,
        covariance=covariance,
        standard_errors=np.sqrt(
            np.maximum(np.diag(covariance), 0.0)
        ),
        success=bool(optimizer_result.success),
        message=str(optimizer_result.message),
        objective_value=float(optimizer_result.fun),
        gbar=gbar,
        omega_gg=omega_result.covariance,
        G=G,
        information=information.value,
        condition_numbers={
            "omega_gg": omega_result.condition_number,
            "information": information.condition_number,
        },
        effective_ranks={
            "omega_gg": omega_result.effective_rank,
            "information": information.effective_rank,
        },
        counts=counts,
        timings=timings,
        regularization=RegularizationInfo(
            omega_gg=omega_result.ridge,
            information=information.ridge,
        ),
        warnings=warnings,
        optimizer_result=optimizer_result,
        reconstruction=reconstruction,
    )


def fit_full_gmm(
    model: MomentModel,
    theta0: Array,
    *,
    weight: Array | None = None,
    bounds: list[tuple[float | None, float | None]] | None = None,
    optimizer_method: str = "L-BFGS-B",
    optimizer_options: dict[str, Any] | None = None,
    reconstruct: bool = False,
    covariance_type: CovarianceType = "iid",
    clusters: Array | None = None,
    ridge: float = 0.0,
    condition_limit: float = 1e12,
) -> GMMResult:
    """Fit two-step efficient GMM using the complete moment system."""

    theta0 = np.asarray(theta0, dtype=float)
    counts = EvaluationCounts()
    timings = StageTimings()
    warnings: list[str] = []

    g_probe = _moments(
        model,
        "tractable_moments",
        theta0,
    )
    counts.tractable_moments += 1

    h_probe = _moments(
        model,
        "demanding_moments",
        theta0,
    )
    counts.demanding_moments_projection += 1

    if g_probe.shape[0] != h_probe.shape[0]:
        raise ModelContractError(
            "Tractable and demanding moments must share "
            "statistical units."
        )

    q = g_probe.shape[1] + h_probe.shape[1]
    stage_one_weight = _validate_weight(
        weight,
        q,
        "weight",
    )

    def stacked_moments(theta: Array) -> Array:
        counts.tractable_moments += 1
        counts.demanding_moments_projection += 1
        g = _moments(
            model,
            "tractable_moments",
            theta,
        )
        h = _moments(
            model,
            "demanding_moments",
            theta,
        )
        if g.shape[0] != h.shape[0]:
            raise ModelContractError(
                "Tractable and demanding moments must share "
                "statistical units."
            )
        return np.column_stack([g, h])

    def criterion(theta: Array, weight_matrix: Array) -> float:
        counts.tractable_objective += 1
        mbar = stacked_moments(theta).mean(axis=0)
        return float(mbar @ weight_matrix @ mbar)

    start = perf_counter()

    stage_one_result = minimize(
        criterion,
        theta0,
        args=(stage_one_weight,),
        method=optimizer_method,
        bounds=bounds,
        options=optimizer_options,
    )
    preliminary = np.asarray(stage_one_result.x, dtype=float)

    moments_stage_one = stacked_moments(preliminary)

    cluster_ids: Array | None
    if covariance_type == "iid":
        cluster_ids = None
    elif covariance_type == "cluster":
        cluster_ids = _clusters_from_model(
            model,
            clusters,
        )
    else:
        raise ValueError(
            "covariance_type must be 'iid' or 'cluster'."
        )

    stage_one_covariance = CovarianceOperator(
        ridge=ridge,
        condition_limit=condition_limit,
    ).fit(
        moments_stage_one,
        covariance_type=covariance_type,
        clusters=cluster_ids,
    )

    stage_two_result = minimize(
        criterion,
        preliminary,
        args=(stage_one_covariance.weight,),
        method=optimizer_method,
        bounds=bounds,
        options=optimizer_options,
    )
    theta = np.asarray(stage_two_result.x, dtype=float)

    timings.localization = perf_counter() - start

    moments_final = stacked_moments(theta)
    g = moments_final[:, :g_probe.shape[1]]
    h = moments_final[:, g_probe.shape[1]:]

    start = perf_counter()

    G = _jacobian(
        model,
        theta,
        counts=counts,
        bounds=bounds,
        fd_rel_step=None,
        demanding=False,
    )

    H = _jacobian(
        model,
        theta,
        counts=counts,
        bounds=bounds,
        fd_rel_step=None,
        demanding=True,
    )

    D = np.vstack([G, H])

    final_covariance = CovarianceOperator(
        ridge=ridge,
        condition_limit=condition_limit,
    ).fit(
        moments_final,
        covariance_type=covariance_type,
        clusters=cluster_ids,
    )

    information = stable_matrix(
        D.T @ final_covariance.weight @ D,
        ridge=ridge,
        condition_limit=condition_limit,
    )
    covariance = solve(
        information.value,
        np.eye(theta.size),
    ) / moments_final.shape[0]

    timings.inference = perf_counter() - start

    reconstruction = None
    if reconstruct:
        start = perf_counter()
        reconstruction = _reconstruct(
            model,
            theta,
            counts,
        )
        timings.reconstruction = perf_counter() - start

    if not stage_one_result.success:
        warnings.append(
            "Full GMM stage-one optimization did not report convergence."
        )

    if not stage_two_result.success:
        warnings.append(
            "Full GMM stage-two optimization did not report convergence."
        )

    if stage_one_covariance.ridge > 0:
        warnings.append(
            "Regularization was applied to the stage-one covariance matrix."
        )

    if final_covariance.ridge > 0:
        warnings.append(
            "Regularization was applied to the final covariance matrix."
        )

    if information.ridge > 0:
        warnings.append(
            "Regularization was applied to the information matrix."
        )

    success = bool(
        stage_one_result.success
        and stage_two_result.success
    )
    message = (
        f"Stage 1: {stage_one_result.message}; "
        f"Stage 2: {stage_two_result.message}"
    )

    return GMMResult(
        method="full_gmm",
        theta=theta,
        preliminary_theta=preliminary,
        covariance=covariance,
        standard_errors=np.sqrt(
            np.maximum(np.diag(covariance), 0.0)
        ),
        success=success,
        message=message,
        objective_value=float(stage_two_result.fun),
        gbar=g.mean(axis=0),
        hbar=h.mean(axis=0),
        G=G,
        H=H,
        information=information.value,
        condition_numbers={
            "omega_stage_one": (
                stage_one_covariance.condition_number
            ),
            "omega_full": final_covariance.condition_number,
            "information": information.condition_number,
        },
        effective_ranks={
            "omega_stage_one": (
                stage_one_covariance.effective_rank
            ),
            "omega_full": final_covariance.effective_rank,
            "information": information.effective_rank,
        },
        counts=counts,
        timings=timings,
        regularization=RegularizationInfo(
            residual_covariance=(
                stage_one_covariance.ridge
            ),
            omega_gg=final_covariance.ridge,
            information=information.ridge,
        ),
        warnings=warnings,
        optimizer_result={
            "stage_one": stage_one_result,
            "stage_two": stage_two_result,
            "stage_one_weight": stage_one_weight,
            "efficient_weight": stage_one_covariance.weight,
        },
        reconstruction=reconstruction,
    )


def fit_seip(
    model: MomentModel,
    theta0: Array,
    *,
    preliminary_theta: Array | None = None,
    tractable_weight: Array | None = None,
    bounds: list[tuple[float | None, float | None]] | None = None,
    covariance_type: CovarianceType = "iid",
    clusters: Array | None = None,
    ridge: float = 0.0,
    condition_limit: float = 1e12,
    fd_rel_step: float | None = None,
    damping: float = 1.0,
    optimizer_method: str = "L-BFGS-B",
    optimizer_options: dict[str, Any] | None = None,
    reconstruct: bool = False,
    orthogonality_tolerance: float = 1e-7,
    relative_update_warning: float = 0.25,
) -> GMMResult:
    """Fit residual-only SOP with matched tractable localization."""

    if not (0.0 < damping <= 1.0):
        raise ValueError("damping must lie in (0, 1].")

    theta0 = np.asarray(theta0, dtype=float)
    counts = EvaluationCounts()
    timings = StageTimings()
    warnings: list[str] = []
    tractable_weight_matrix: Array | None = None

    probe = _moments(
        model,
        "tractable_moments",
        theta0,
    )
    counts.tractable_moments += 1
    q_g = probe.shape[1]

    if covariance_type == "iid":
        localization_cluster_ids = None
    elif covariance_type == "cluster":
        localization_cluster_ids = _clusters_from_model(
            model,
            clusters,
        )
    else:
        raise ValueError(
            "covariance_type must be 'iid' or 'cluster'."
        )

    stage_one_result = None
    stage_two_result = None
    stage_one_covariance = None
    stage_one_weight = None

    if preliminary_theta is not None:
        # Backward-compatible expert path: the supplied value is treated as
        # the final tractable localizer. The supplied tractable weight should
        # be the fixed weight used to construct that estimator. If omitted,
        # the historical identity-weight convention is retained.
        preliminary = np.asarray(
            preliminary_theta,
            dtype=float,
        )

        if preliminary.shape != theta0.shape:
            raise ModelContractError(
                "preliminary_theta and theta0 must have "
                "the same shape."
            )

        initial_tractable = preliminary.copy()
        tractable_weight_matrix = _validate_weight(
            tractable_weight,
            q_g,
            "tractable_weight",
        )
        success = True
        message = "User-supplied final tractable estimator."
        objective_value = None
        optimizer_result = None
    else:
        localization_start = perf_counter()

        if tractable_weight is None:
            # Stage 1: inexpensive identity-weight localization.
            stage_one_weight = _identity_weight(q_g)

            def stage_one_tractable_objective(theta: Array) -> float:
                counts.tractable_objective += 1
                counts.tractable_moments += 1
                gbar_stage_one = _mean_moments(
                    model,
                    "tractable_moments",
                    theta,
                )
                return float(
                    gbar_stage_one
                    @ stage_one_weight
                    @ gbar_stage_one
                )

            stage_one_result = minimize(
                stage_one_tractable_objective,
                theta0,
                method=optimizer_method,
                bounds=bounds,
                options=optimizer_options,
            )
            initial_tractable = np.asarray(
                stage_one_result.x,
                dtype=float,
            )

            g_stage_one = _moments(
                model,
                "tractable_moments",
                initial_tractable,
            )
            counts.tractable_moments += 1

            stage_one_covariance = CovarianceOperator(
                ridge=ridge,
                condition_limit=condition_limit,
            ).fit(
                g_stage_one,
                covariance_type=covariance_type,
                clusters=localization_cluster_ids,
            )
            tractable_weight_matrix = stage_one_covariance.weight
        else:
            # A user-supplied weight is already fixed before the final
            # tractable optimization, so no preliminary weighting stage is
            # required.
            initial_tractable = theta0.copy()
            tractable_weight_matrix = _validate_weight(
                tractable_weight,
                q_g,
                "tractable_weight",
            )

        # Stage 2: final tractable optimization with W_g held fixed. This is
        # the estimator denoted by tilde-theta in Section 3.
        def stage_two_tractable_objective(theta: Array) -> float:
            counts.tractable_objective += 1
            counts.tractable_moments += 1
            gbar_stage_two = _mean_moments(
                model,
                "tractable_moments",
                theta,
            )
            return float(
                gbar_stage_two
                @ tractable_weight_matrix
                @ gbar_stage_two
            )

        stage_two_result = minimize(
            stage_two_tractable_objective,
            initial_tractable,
            method=optimizer_method,
            bounds=bounds,
            options=optimizer_options,
        )
        preliminary = np.asarray(
            stage_two_result.x,
            dtype=float,
        )
        timings.localization = perf_counter() - localization_start

        stage_one_success = (
            True
            if stage_one_result is None
            else bool(stage_one_result.success)
        )
        success = bool(
            stage_one_success
            and stage_two_result.success
        )

        if stage_one_result is None:
            stage_one_message = "supplied fixed tractable weight"
        else:
            stage_one_message = str(stage_one_result.message)

        message = (
            f"Tractable stage 1: {stage_one_message}; "
            f"tractable stage 2: {stage_two_result.message}"
        )
        objective_value = float(stage_two_result.fun)
        optimizer_result = {
            "stage_one": stage_one_result,
            "stage_two": stage_two_result,
            "stage_one_weight": stage_one_weight,
            "tractable_weight": tractable_weight_matrix,
            "tractable_weight_covariance": stage_one_covariance,
        }

        if stage_one_result is not None and not stage_one_result.success:
            warnings.append(
                "Tractable stage-one optimization did not report "
                "convergence."
            )

        if not stage_two_result.success:
            warnings.append(
                "Final matched-weight tractable optimization did not "
                "report convergence."
            )

        if (
            stage_one_covariance is not None
            and stage_one_covariance.ridge > 0
        ):
            warnings.append(
                "Regularization was applied when constructing the "
                "tractable weighting matrix."
            )

    start = perf_counter()

    g = _moments(
        model,
        "tractable_moments",
        preliminary,
    )
    counts.tractable_moments += 1

    if tractable_weight_matrix is None:
        tractable_weight_matrix = _validate_weight(
            tractable_weight,
            g.shape[1],
            "tractable_weight",
        )

    h = _moments(
        model,
        "demanding_moments",
        preliminary,
    )
    counts.demanding_moments_projection += 1

    timings.moments = perf_counter() - start

    if g.shape[0] != h.shape[0]:
        raise ModelContractError(
            "Tractable and demanding moments must share "
            "statistical units."
        )

    n = g.shape[0]

    start = perf_counter()

    G = _jacobian(
        model,
        preliminary,
        counts=counts,
        bounds=bounds,
        fd_rel_step=fd_rel_step,
        demanding=False,
    )

    H = _jacobian(
        model,
        preliminary,
        counts=counts,
        bounds=bounds,
        fd_rel_step=fd_rel_step,
        demanding=True,
    )

    timings.derivatives = perf_counter() - start

    start = perf_counter()

    if covariance_type == "iid":
        cluster_ids = None
    elif covariance_type == "cluster":
        cluster_ids = _clusters_from_model(
            model,
            clusters,
        )
    else:
        raise ValueError(
            "covariance_type must be 'iid' or 'cluster'."
        )

    projection_operator = OrthogonalProjection(
        ridge=ridge,
        condition_limit=condition_limit,
    )

    projection_result = projection_operator.fit(
        g=g,
        h=h,
        G=G,
        H=H,
        covariance_type=covariance_type,
        clusters=cluster_ids,
        tractable_weight=tractable_weight_matrix,
    )

    coefficient = projection_result.coefficient
    residuals = projection_result.residuals
    residual_covariance = (
        projection_result.residual_covariance
    )
    residualized_jacobian = (
        projection_result.residualized_jacobian
    )
    information = projection_result.information
    tractable_score = projection_result.tractable_score
    residual_score = projection_result.residual_score
    projected_score = projection_result.projected_score
    orthogonality = (
        projection_result.orthogonality_residual
    )

    gbar = g.mean(axis=0)
    hbar = h.mean(axis=0)
    nubar = residuals.mean(axis=0)

    # Diagnostic update using the complete transformed score.
    full_score_update = -solve(
        information,
        projected_score,
    )

    # Canonical Section 3 residual-only SOP update.
    residual_only_update = -solve(
        information,
        residual_score,
    )

    raw_update = residual_only_update
    update = damping * raw_update
    theta = preliminary + update

    tractable_foc_norm = float(
        np.linalg.norm(tractable_score)
    )
    update_difference_norm = float(
        np.linalg.norm(
            full_score_update
            - residual_only_update
        )
    )

    timings.projection = perf_counter() - start

    if bounds is not None:
        for j, (lower, upper) in enumerate(bounds):
            if lower is not None and theta[j] < lower:
                warnings.append(
                    "Final update violates lower bound "
                    f"for parameter {j}."
                )

            if upper is not None and theta[j] > upper:
                warnings.append(
                    "Final update violates upper bound "
                    f"for parameter {j}."
                )

    orthogonality_norm = float(
        np.linalg.norm(orthogonality)
    )

    if orthogonality_norm > orthogonality_tolerance:
        warnings.append(
            "Projection orthogonality residual "
            f"{orthogonality_norm:.3g} exceeds tolerance "
            f"{orthogonality_tolerance:.3g}."
        )

    relative_update = float(
        np.linalg.norm(update)
        / (1.0 + np.linalg.norm(preliminary))
    )

    if relative_update > relative_update_warning:
        warnings.append(
            "One-step correction is large relative to "
            f"preliminary estimate ({relative_update:.3g})."
        )

    if any(
        level > 0
        for level in projection_result.ridge_levels.values()
    ):
        warnings.append(
            "Regularization was applied to at least one matrix."
        )

    start = perf_counter()

    covariance = solve(
        information,
        np.eye(theta.size),
    ) / n
    standard_errors = np.sqrt(
        np.maximum(np.diag(covariance), 0.0)
    )

    timings.inference = perf_counter() - start

    reconstruction = None
    if reconstruct:
        start = perf_counter()
        reconstruction = _reconstruct(
            model,
            theta,
            counts,
        )
        timings.reconstruction = perf_counter() - start

    return GMMResult(
        method="seip",
        theta=theta,
        preliminary_theta=preliminary,
        initial_tractable_theta=initial_tractable,
        covariance=covariance,
        standard_errors=standard_errors,
        success=success,
        message=message,
        objective_value=objective_value,
        update=update,
        raw_update=raw_update,
        full_score_update=full_score_update,
        residual_only_update=residual_only_update,
        tractable_score=tractable_score,
        residual_score=residual_score,
        tractable_weight=tractable_weight_matrix,
        tractable_foc_norm=tractable_foc_norm,
        update_difference_norm=update_difference_norm,
        damping_factor=damping,
        gbar=gbar,
        hbar=hbar,
        nubar=nubar,
        omega_gg=projection_result.omega_gg,
        omega_hg=projection_result.omega_hg,
        residual_covariance=residual_covariance,
        projection=coefficient,
        G=G,
        H=H,
        R=residualized_jacobian,
        information=information,
        orthogonality_residual=orthogonality,
        condition_numbers=(
            projection_result.condition_numbers
        ),
        effective_ranks=projection_result.effective_ranks,
        counts=counts,
        timings=timings,
        regularization=RegularizationInfo(
            omega_gg=(
                projection_result.ridge_levels["omega_gg"]
            ),
            residual_covariance=(
                projection_result.ridge_levels[
                    "residual_covariance"
                ]
            ),
            information=(
                projection_result.ridge_levels["information"]
            ),
        ),
        warnings=warnings,
        optimizer_result=optimizer_result,
        reconstruction=reconstruction,
    )

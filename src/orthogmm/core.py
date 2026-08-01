from __future__ import annotations

from time import perf_counter
from typing import Any, Literal

import numpy as np
from scipy.optimize import minimize

from .exceptions import ModelContractError
from .jacobians import FallbackJacobian, JacobianContext
from .linalg import solve, stable_matrix
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
    """Construct a tractable or demanding Jacobian.

    The default strategy uses a model-supplied analytical Jacobian when
    available and otherwise falls back to finite differences of mean moments.
    """
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

    omega = np.atleast_2d(
        np.cov(g, rowvar=False, bias=True)
    )
    omega_stable = stable_matrix(omega)
    information = stable_matrix(
        G.T @ solve(omega_stable.value, G)
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
        omega_gg=omega_stable.value,
        G=G,
        information=information.value,
        condition_numbers={
            "omega_gg": omega_stable.condition_number,
            "information": information.condition_number,
        },
        effective_ranks={
            "omega_gg": omega_stable.effective_rank,
            "information": information.effective_rank,
        },
        counts=counts,
        timings=timings,
        regularization=RegularizationInfo(
            omega_gg=omega_stable.ridge,
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
) -> GMMResult:
    theta0 = np.asarray(theta0, dtype=float)
    counts = EvaluationCounts()
    timings = StageTimings()

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
    weight_matrix = _validate_weight(weight, q, "weight")

    def objective(theta: Array) -> float:
        counts.tractable_objective += 1
        counts.tractable_moments += 1
        counts.demanding_moments_projection += 1

        mbar = np.concatenate(
            [
                _mean_moments(
                    model,
                    "tractable_moments",
                    theta,
                ),
                _mean_moments(
                    model,
                    "demanding_moments",
                    theta,
                ),
            ]
        )

        return float(mbar @ weight_matrix @ mbar)

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

    h = _moments(model, "demanding_moments", theta)
    counts.demanding_moments_projection += 1

    m = np.column_stack([g, h])

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

    omega = np.atleast_2d(
        np.cov(m, rowvar=False, bias=True)
    )
    omega_stable = stable_matrix(omega)
    information = stable_matrix(
        D.T @ solve(omega_stable.value, D)
    )
    covariance = solve(
        information.value,
        np.eye(theta.size),
    ) / m.shape[0]

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
            "Full GMM optimization did not report convergence."
        )

    return GMMResult(
        method="full_gmm",
        theta=theta,
        preliminary_theta=theta,
        covariance=covariance,
        standard_errors=np.sqrt(
            np.maximum(np.diag(covariance), 0.0)
        ),
        success=bool(optimizer_result.success),
        message=str(optimizer_result.message),
        objective_value=float(optimizer_result.fun),
        gbar=g.mean(axis=0),
        hbar=h.mean(axis=0),
        G=G,
        H=H,
        information=information.value,
        condition_numbers={
            "omega_full": omega_stable.condition_number,
            "information": information.condition_number,
        },
        effective_ranks={
            "omega_full": omega_stable.effective_rank,
            "information": information.effective_rank,
        },
        counts=counts,
        timings=timings,
        regularization=RegularizationInfo(
            information=information.ridge,
        ),
        warnings=warnings,
        optimizer_result=optimizer_result,
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
    """Fit the canonical projected one-step estimator."""

    if not (0.0 < damping <= 1.0):
        raise ValueError("damping must lie in (0, 1].")

    theta0 = np.asarray(theta0, dtype=float)
    counts = EvaluationCounts()
    timings = StageTimings()
    warnings: list[str] = []

    if preliminary_theta is None:
        probe = _moments(
            model,
            "tractable_moments",
            theta0,
        )
        counts.tractable_moments += 1

        tractable_weight_matrix = _validate_weight(
            tractable_weight,
            probe.shape[1],
            "tractable_weight",
        )

        def objective(theta: Array) -> float:
            counts.tractable_objective += 1
            counts.tractable_moments += 1

            gbar = _mean_moments(
                model,
                "tractable_moments",
                theta,
            )

            return float(
                gbar @ tractable_weight_matrix @ gbar
            )

        start = perf_counter()
        optimizer_result = minimize(
            objective,
            theta0,
            method=optimizer_method,
            bounds=bounds,
            options=optimizer_options,
        )
        timings.localization = perf_counter() - start

        preliminary = np.asarray(
            optimizer_result.x,
            dtype=float,
        )
        success = bool(optimizer_result.success)
        message = str(optimizer_result.message)
        objective_value = float(optimizer_result.fun)

        if not success:
            warnings.append(
                "Tractable localization did not report convergence."
            )
    else:
        preliminary = np.asarray(
            preliminary_theta,
            dtype=float,
        )

        if preliminary.shape != theta0.shape:
            raise ModelContractError(
                "preliminary_theta and theta0 must have "
                "the same shape."
            )

        optimizer_result = None
        success = True
        message = "User-supplied preliminary estimator."
        objective_value = None

    start = perf_counter()

    g = _moments(
        model,
        "tractable_moments",
        preliminary,
    )
    counts.tractable_moments += 1

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
    projected_score = projection_result.projected_score
    orthogonality = (
        projection_result.orthogonality_residual
    )

    gbar = g.mean(axis=0)
    hbar = h.mean(axis=0)
    nubar = residuals.mean(axis=0)

    update = -damping * solve(
        information,
        projected_score,
    )
    theta = preliminary + update

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
        covariance=covariance,
        standard_errors=standard_errors,
        success=success,
        message=message,
        objective_value=objective_value,
        update=update,
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

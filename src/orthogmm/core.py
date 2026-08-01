from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Any, Literal

import numpy as np
from scipy.optimize import minimize

from .differentiation import finite_difference_jacobian
from .exceptions import ModelContractError, NumericalError
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
        raise ModelContractError(f"{name} must return a unit-by-moment array.")
    if not np.all(np.isfinite(out)):
        raise ModelContractError(f"{name} returned non-finite values.")
    return out


def _mean_moments(model: MomentModel, name: str, theta: Array) -> Array:
    return _moments(model, name, theta).mean(axis=0)


def _identity_weight(q: int) -> Array:
    return np.eye(q, dtype=float)


def _validate_weight(w: Array | None, q: int, name: str) -> Array:
    if w is None:
        return _identity_weight(q)
    w = np.asarray(w, dtype=float)
    if w.shape != (q, q):
        raise ModelContractError(f"{name} must have shape ({q}, {q}).")
    return stable_matrix(w).value


def _jacobian(
    model: MomentModel,
    method_name: str,
    moment_name: str,
    theta: Array,
    *,
    counts: EvaluationCounts,
    bounds: list[tuple[float | None, float | None]] | None,
    fd_rel_step: float | None,
    demanding: bool,
) -> Array:
    method = getattr(model, method_name, None)
    if callable(method):
        try:
            if demanding:
                counts.demanding_jacobian += 1
            else:
                counts.tractable_jacobian += 1
            jac = np.asarray(method(theta), dtype=float)
        except NotImplementedError:
            method = None
    if not callable(method):
        def mean_fn(x: Array) -> Array:
            if demanding:
                counts.demanding_moments_derivative += 1
            else:
                counts.tractable_moments += 1
            return _mean_moments(model, moment_name, x)
        jac = finite_difference_jacobian(
            mean_fn, theta, rel_step=fd_rel_step, bounds=bounds
        )
    if jac.ndim != 2 or jac.shape[1] != theta.size:
        raise ModelContractError(
            f"{method_name} must return a moment-by-parameter matrix with {theta.size} columns."
        )
    if not np.all(np.isfinite(jac)):
        raise ModelContractError(f"{method_name} returned non-finite values.")
    return jac


def _reconstruct(model: MomentModel, theta: Array, counts: EvaluationCounts) -> Any:
    fn = getattr(model, "reconstruct", None)
    if callable(fn):
        counts.reconstruction += 1
        return fn(theta)
    return None


def _clusters_from_model(model: MomentModel, clusters: Array | None) -> Array:
    if clusters is not None:
        return np.asarray(clusters)
    fn = getattr(model, "unit_ids", None)
    if callable(fn):
        return np.asarray(fn())
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
    w = _validate_weight(weight, probe.shape[1], "weight")

    def objective(theta: Array) -> float:
        counts.tractable_objective += 1
        counts.tractable_moments += 1
        gbar = _mean_moments(model, "tractable_moments", theta)
        return float(gbar @ w @ gbar)

    start = perf_counter()
    opt = minimize(
        objective,
        theta0,
        method=optimizer_method,
        bounds=bounds,
        options=optimizer_options,
    )
    timings.localization = perf_counter() - start
    theta = np.asarray(opt.x, dtype=float)
    g = _moments(model, "tractable_moments", theta)
    counts.tractable_moments += 1
    gbar = g.mean(axis=0)

    start = perf_counter()
    G = _jacobian(
        model, "tractable_jacobian", "tractable_moments", theta,
        counts=counts, bounds=bounds, fd_rel_step=None, demanding=False,
    )
    omega = np.cov(g, rowvar=False, bias=True)
    omega = np.atleast_2d(omega)
    omega_stable = stable_matrix(omega)
    info = stable_matrix(G.T @ solve(omega_stable.value, G))
    covariance = solve(info.value, np.eye(theta.size)) / g.shape[0]
    timings.inference = perf_counter() - start

    reconstruction = None
    if reconstruct:
        start = perf_counter()
        reconstruction = _reconstruct(model, theta, counts)
        timings.reconstruction = perf_counter() - start

    warnings: list[str] = []
    if not opt.success:
        warnings.append("Tractable localization did not report convergence.")

    return GMMResult(
        method="tractable_gmm",
        theta=theta,
        preliminary_theta=theta,
        covariance=covariance,
        standard_errors=np.sqrt(np.maximum(np.diag(covariance), 0.0)),
        success=bool(opt.success),
        message=str(opt.message),
        objective_value=float(opt.fun),
        gbar=gbar,
        omega_gg=omega_stable.value,
        G=G,
        information=info.value,
        condition_numbers={"omega_gg": omega_stable.condition_number, "information": info.condition_number},
        effective_ranks={"omega_gg": omega_stable.effective_rank, "information": info.effective_rank},
        counts=counts,
        timings=timings,
        regularization=RegularizationInfo(omega_gg=omega_stable.ridge, information=info.ridge),
        warnings=warnings,
        optimizer_result=opt,
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
    gp = _moments(model, "tractable_moments", theta0); counts.tractable_moments += 1
    hp = _moments(model, "demanding_moments", theta0); counts.demanding_moments_projection += 1
    if gp.shape[0] != hp.shape[0]:
        raise ModelContractError("Tractable and demanding moments must share statistical units.")
    q = gp.shape[1] + hp.shape[1]
    w = _validate_weight(weight, q, "weight")

    def objective(theta: Array) -> float:
        counts.tractable_objective += 1
        counts.tractable_moments += 1
        counts.demanding_moments_projection += 1
        mbar = np.concatenate([
            _mean_moments(model, "tractable_moments", theta),
            _mean_moments(model, "demanding_moments", theta),
        ])
        return float(mbar @ w @ mbar)

    start = perf_counter()
    opt = minimize(objective, theta0, method=optimizer_method, bounds=bounds, options=optimizer_options)
    timings.localization = perf_counter() - start
    theta = np.asarray(opt.x, dtype=float)
    g = _moments(model, "tractable_moments", theta); counts.tractable_moments += 1
    h = _moments(model, "demanding_moments", theta); counts.demanding_moments_projection += 1
    m = np.column_stack([g, h])
    mbar = m.mean(axis=0)

    start = perf_counter()
    G = _jacobian(model, "tractable_jacobian", "tractable_moments", theta,
                  counts=counts, bounds=bounds, fd_rel_step=None, demanding=False)
    H = _jacobian(model, "demanding_jacobian", "demanding_moments", theta,
                  counts=counts, bounds=bounds, fd_rel_step=None, demanding=True)
    D = np.vstack([G, H])
    omega = np.cov(m, rowvar=False, bias=True)
    omega = np.atleast_2d(omega)
    omega_stable = stable_matrix(omega)
    info = stable_matrix(D.T @ solve(omega_stable.value, D))
    covariance = solve(info.value, np.eye(theta.size)) / m.shape[0]
    timings.inference = perf_counter() - start

    reconstruction = None
    if reconstruct:
        start = perf_counter(); reconstruction = _reconstruct(model, theta, counts); timings.reconstruction = perf_counter() - start

    warnings = [] if opt.success else ["Full GMM optimization did not report convergence."]
    return GMMResult(
        method="full_gmm", theta=theta, preliminary_theta=theta,
        covariance=covariance,
        standard_errors=np.sqrt(np.maximum(np.diag(covariance), 0.0)),
        success=bool(opt.success), message=str(opt.message), objective_value=float(opt.fun),
        gbar=g.mean(axis=0), hbar=h.mean(axis=0), G=G, H=H, information=info.value,
        condition_numbers={"omega_full": omega_stable.condition_number, "information": info.condition_number},
        effective_ranks={"omega_full": omega_stable.effective_rank, "information": info.effective_rank},
        counts=counts, timings=timings,
        regularization=RegularizationInfo(information=info.ridge),
        warnings=warnings, optimizer_result=opt, reconstruction=reconstruction,
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
    """Fit the canonical projected one-step estimator from Appendix C."""
    if not (0.0 < damping <= 1.0):
        raise ValueError("damping must lie in (0, 1].")
    theta0 = np.asarray(theta0, dtype=float)
    counts = EvaluationCounts()
    timings = StageTimings()
    warnings: list[str] = []

    if preliminary_theta is None:
        probe = _moments(model, "tractable_moments", theta0); counts.tractable_moments += 1
        wg = _validate_weight(tractable_weight, probe.shape[1], "tractable_weight")

        def objective(theta: Array) -> float:
            counts.tractable_objective += 1
            counts.tractable_moments += 1
            gbar = _mean_moments(model, "tractable_moments", theta)
            return float(gbar @ wg @ gbar)

        start = perf_counter()
        opt = minimize(objective, theta0, method=optimizer_method, bounds=bounds, options=optimizer_options)
        timings.localization = perf_counter() - start
        preliminary = np.asarray(opt.x, dtype=float)
        success = bool(opt.success)
        message = str(opt.message)
        objective_value = float(opt.fun)
        if not success:
            warnings.append("Tractable localization did not report convergence.")
    else:
        preliminary = np.asarray(preliminary_theta, dtype=float)
        if preliminary.shape != theta0.shape:
            raise ModelContractError("preliminary_theta and theta0 must have the same shape.")
        opt = None
        success = True
        message = "User-supplied preliminary estimator."
        objective_value = None

    start = perf_counter()
    g = _moments(model, "tractable_moments", preliminary); counts.tractable_moments += 1
    h = _moments(model, "demanding_moments", preliminary); counts.demanding_moments_projection += 1
    timings.moments = perf_counter() - start
    if g.shape[0] != h.shape[0]:
        raise ModelContractError("Tractable and demanding moments must share statistical units.")
    n = g.shape[0]

    start = perf_counter()
    G = _jacobian(model, "tractable_jacobian", "tractable_moments", preliminary,
                  counts=counts, bounds=bounds, fd_rel_step=fd_rel_step, demanding=False)
    H = _jacobian(model, "demanding_jacobian", "demanding_moments", preliminary,
                  counts=counts, bounds=bounds, fd_rel_step=fd_rel_step, demanding=True)
    timings.derivatives = perf_counter() - start

    start = perf_counter()
    if covariance_type == "iid":
        cluster_ids = None
    elif covariance_type == "cluster":
        cluster_ids = _clusters_from_model(model, clusters)
    else:
        raise ValueError("covariance_type must be 'iid' or 'cluster'.")

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

    B = projection_result.coefficient
    nu = projection_result.residuals
    S_value = projection_result.residual_covariance
    R = projection_result.residualized_jacobian
    J_value = projection_result.information
    psi = projection_result.projected_score
    orthogonality = projection_result.orthogonality_residual
    omega_gg_value = projection_result.omega_gg
    omega_hg = projection_result.omega_hg

    gbar = g.mean(axis=0)
    hbar = h.mean(axis=0)
    nubar = nu.mean(axis=0)

    update = -damping * solve(J_value, psi)
    theta = preliminary + update
    timings.projection = perf_counter() - start

    if bounds is not None:
        for j, (lower, upper) in enumerate(bounds):
            if lower is not None and theta[j] < lower:
                warnings.append(f"Final update violates lower bound for parameter {j}.")
            if upper is not None and theta[j] > upper:
                warnings.append(f"Final update violates upper bound for parameter {j}.")

    ortho_norm = float(np.linalg.norm(orthogonality))
    if ortho_norm > orthogonality_tolerance:
        warnings.append(
            f"Projection orthogonality residual {ortho_norm:.3g} exceeds tolerance {orthogonality_tolerance:.3g}."
        )
    relative_update = float(np.linalg.norm(update) / (1.0 + np.linalg.norm(preliminary)))
    if relative_update > relative_update_warning:
        warnings.append(
            f"One-step correction is large relative to preliminary estimate ({relative_update:.3g})."
        )
    if any(level > 0 for level in projection_result.ridge_levels.values()):
        warnings.append("Regularization was applied to at least one matrix.")

    start = perf_counter()
    covariance = solve(J_value, np.eye(theta.size)) / n
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    timings.inference = perf_counter() - start

    reconstruction = None
    if reconstruct:
        start = perf_counter()
        reconstruction = _reconstruct(model, theta, counts)
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
        omega_gg=omega_gg_value,
        omega_hg=omega_hg,
        residual_covariance=S_value,
        projection=B,
        G=G,
        H=H,
        R=R,
        information=J_value,
        orthogonality_residual=orthogonality,
        condition_numbers=projection_result.condition_numbers,
        effective_ranks=projection_result.effective_ranks,
        counts=counts,
        timings=timings,
        regularization=RegularizationInfo(
            omega_gg=projection_result.ridge_levels["omega_gg"],
            residual_covariance=(
                projection_result.ridge_levels["residual_covariance"]
            ),
            information=projection_result.ridge_levels["information"],
        ),
        warnings=warnings,
        optimizer_result=opt,
        reconstruction=reconstruction,
    )

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .exceptions import ModelContractError
from .types import Array


def finite_difference_jacobian(
    mean_function: Callable[[Array], Array],
    theta: Array,
    *,
    rel_step: float | None = None,
    bounds: list[tuple[float | None, float | None]] | None = None,
) -> Array:
    """Scale-aware finite-difference Jacobian of a vector-valued mean function."""
    theta = np.asarray(theta, dtype=float)
    f0 = np.asarray(mean_function(theta), dtype=float).reshape(-1)
    q, p = f0.size, theta.size
    jac = np.empty((q, p), dtype=float)
    eps_step = np.cbrt(np.finfo(float).eps) if rel_step is None else float(rel_step)
    bounds = bounds or [(None, None)] * p
    if len(bounds) != p:
        raise ModelContractError("bounds must have one pair per parameter.")

    for j in range(p):
        delta = eps_step * max(1.0, abs(theta[j]))
        lower, upper = bounds[j]
        can_minus = lower is None or theta[j] - delta >= lower
        can_plus = upper is None or theta[j] + delta <= upper
        if can_minus and can_plus:
            plus = theta.copy(); plus[j] += delta
            minus = theta.copy(); minus[j] -= delta
            jac[:, j] = (np.asarray(mean_function(plus)).reshape(-1) -
                         np.asarray(mean_function(minus)).reshape(-1)) / (2.0 * delta)
        elif can_plus:
            plus = theta.copy(); plus[j] += delta
            jac[:, j] = (np.asarray(mean_function(plus)).reshape(-1) - f0) / delta
        elif can_minus:
            minus = theta.copy(); minus[j] -= delta
            jac[:, j] = (f0 - np.asarray(mean_function(minus)).reshape(-1)) / delta
        else:
            raise ModelContractError(f"No feasible finite-difference step for parameter {j}.")

    if not np.all(np.isfinite(jac)):
        raise ModelContractError("Finite-difference Jacobian contains non-finite values.")
    return jac

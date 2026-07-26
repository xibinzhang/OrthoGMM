from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .exceptions import NumericalError
from .types import Array


@dataclass(slots=True)
class StableMatrix:
    value: Array
    ridge: float
    condition_number: float
    effective_rank: int


def symmetrize(a: Array) -> Array:
    return 0.5 * (a + a.T)


def stable_matrix(
    a: Array,
    *,
    ridge: float = 0.0,
    condition_limit: float = 1e12,
    eigenvalue_floor: float = 1e-10,
) -> StableMatrix:
    """Return a symmetric, numerically stable version of a square matrix."""
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise NumericalError("Expected a square matrix.")
    if not np.all(np.isfinite(a)):
        raise NumericalError("Matrix contains non-finite values.")

    b = symmetrize(a)
    evals = np.linalg.eigvalsh(b)
    scale = max(1.0, float(np.max(np.abs(evals))))
    min_allowed = eigenvalue_floor * scale
    required = max(0.0, min_allowed - float(np.min(evals)))
    applied = max(float(ridge), required)
    if applied > 0:
        b = b + applied * np.eye(b.shape[0])

    cond = float(np.linalg.cond(b))
    if not np.isfinite(cond) or cond > condition_limit:
        # Add enough ridge iteratively to avoid hidden pseudoinverse behavior.
        trial = max(applied, min_allowed)
        for _ in range(10):
            candidate = b + trial * np.eye(b.shape[0])
            cond_candidate = float(np.linalg.cond(candidate))
            if np.isfinite(cond_candidate) and cond_candidate <= condition_limit:
                b = candidate
                applied += trial
                cond = cond_candidate
                break
            trial *= 10.0
        else:
            raise NumericalError("Could not regularize matrix to a stable condition number.")

    rank = int(np.linalg.matrix_rank(b))
    return StableMatrix(b, applied, cond, rank)


def solve(a: Array, b: Array) -> Array:
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError as exc:
        raise NumericalError("Linear system is singular or ill-conditioned.") from exc

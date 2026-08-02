"""Stable functional API for OrthoGMM.

The concise function names in this module are the preferred public interface.
The original ``*_gmm`` and ``fit_seip`` names remain available for backward
compatibility.
"""

from __future__ import annotations

from typing import Any

from .core import fit_full_gmm, fit_seip, fit_tractable_gmm
from .types import Array, GMMResult, MomentModel


def fit_tractable(
    model: MomentModel,
    theta0: Array,
    **kwargs: Any,
) -> GMMResult:
    """Fit GMM using only the tractable moment subsystem."""

    return fit_tractable_gmm(model, theta0, **kwargs)


def fit_full(
    model: MomentModel,
    theta0: Array,
    **kwargs: Any,
) -> GMMResult:
    """Fit conventional efficient GMM using the full moment system."""

    return fit_full_gmm(model, theta0, **kwargs)


def fit_projection(
    model: MomentModel,
    theta0: Array,
    **kwargs: Any,
) -> GMMResult:
    """Fit Sequential Oracle Projection GMM.

    This is the preferred public name for the estimator implemented by the
    backward-compatible :func:`orthogmm.fit_seip` function.
    """

    return fit_seip(model, theta0, **kwargs)


__all__ = ["fit_full", "fit_projection", "fit_tractable"]

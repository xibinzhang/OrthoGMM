"""Model-independent statistical operators."""

from .base import BaseOperator
from .covariance import (
    CovarianceOperator,
    CovarianceResult,
    CovarianceType,
)
from .projection import OrthogonalProjection, ProjectionResult

__all__ = [
    "BaseOperator",
    "CovarianceOperator",
    "CovarianceResult",
    "CovarianceType",
    "OrthogonalProjection",
    "ProjectionResult",
]

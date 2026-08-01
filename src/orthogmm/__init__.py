"""OrthoGMM: efficient GMM under computational heterogeneity."""

from .blp import FidelityConfig, MultiFidelityBLPModel
from .core import fit_full_gmm, fit_seip, fit_tractable_gmm
from .estimators import BaseEstimator, FullGMM, SEIPEstimator, SOPEstimator, TractableGMM
from .exceptions import ModelContractError, NumericalError, OrthoGMMError
from .model import BaseMomentModel
from .operators import OrthogonalProjection, ProjectionResult
from .types import EvaluationCounts, GMMResult, MomentModel, RegularizationInfo, StageTimings

__all__ = [
    "BaseEstimator",
    "BaseMomentModel",
    "EvaluationCounts",
    "FidelityConfig",
    "FullGMM",
    "GMMResult",
    "ModelContractError",
    "MomentModel",
    "MultiFidelityBLPModel",
    "NumericalError",
    "OrthogonalProjection",
    "OrthoGMMError",
    "ProjectionResult",
    "RegularizationInfo",
    "SEIPEstimator",
    "SOPEstimator",
    "StageTimings",
    "TractableGMM",
    "fit_full_gmm",
    "fit_seip",
    "fit_tractable_gmm",
]

__version__ = "0.2.0.dev0"

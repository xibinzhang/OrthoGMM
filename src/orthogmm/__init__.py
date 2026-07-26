"""OrthoGMM: orthogonal projection GMM under computational heterogeneity."""

from .blp import FidelityConfig, MultiFidelityBLPModel
from .core import fit_full_gmm, fit_seip, fit_tractable_gmm
from .exceptions import ModelContractError, NumericalError, OrthoGMMError
from .types import EvaluationCounts, GMMResult, MomentModel, RegularizationInfo, StageTimings

__all__ = [
    "EvaluationCounts",
    "FidelityConfig",
    "GMMResult",
    "ModelContractError",
    "MomentModel",
    "MultiFidelityBLPModel",
    "NumericalError",
    "OrthoGMMError",
    "RegularizationInfo",
    "StageTimings",
    "fit_full_gmm",
    "fit_seip",
    "fit_tractable_gmm",
]

__version__ = "0.1.0"

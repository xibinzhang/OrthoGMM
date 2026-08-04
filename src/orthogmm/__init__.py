"""OrthoGMM: efficient GMM under computational heterogeneity."""

from .api import fit_full, fit_projection, fit_tractable
from .blp import FidelityConfig, MultiFidelityBLPModel
from .core import fit_full_gmm, fit_seip, fit_tractable_gmm
from .estimators import (
    BaseEstimator,
    FullGMM,
    SEIPEstimator,
    SOPEstimator,
    TractableGMM,
)
from .exceptions import (
    ModelContractError,
    NumericalError,
    OrthoGMMError,
)
from .model import (
    BaseMomentModel,
    RandomCoefficientIntegration,
    RandomCoefficientMomentModel,
)
from .operators import (
    BaseOperator,
    CovarianceOperator,
    CovarianceResult,
    OrthogonalProjection,
    ProjectionResult,
)
from .simulation import (
    ComparisonRecord,
    ComparisonSummary,
    EstimatorComparison,
    GridBenchmark,
    GridCellResult,
    GridComparisonCell,
    GridEstimatorComparison,
    GridParameterSummary,
    GridResults,
    MonteCarloBenchmark,
)
from .types import (
    EvaluationCounts,
    GMMResult,
    MomentModel,
    RegularizationInfo,
    StageTimings,
)

__all__ = [
    "BaseEstimator",
    "BaseMomentModel",
    "BaseOperator",
    "CovarianceOperator",
    "ComparisonRecord",
    "ComparisonSummary",
    "CovarianceResult",
    "EstimatorComparison",
    "EvaluationCounts",
    "FidelityConfig",
    "FullGMM",
    "GridBenchmark",
    "GridCellResult",
    "GridComparisonCell",
    "GridEstimatorComparison",
    "GridParameterSummary",
    "GridResults",
    "GMMResult",
    "ModelContractError",
    "MomentModel",
    "MonteCarloBenchmark",
    "MultiFidelityBLPModel",
    "NumericalError",
    "OrthogonalProjection",
    "OrthoGMMError",
    "ProjectionResult",
    "RandomCoefficientIntegration",
    "RandomCoefficientMomentModel",
    "RegularizationInfo",
    "SEIPEstimator",
    "SOPEstimator",
    "StageTimings",
    "TractableGMM",
    "fit_full",
    "fit_full_gmm",
    "fit_projection",
    "fit_seip",
    "fit_tractable",
    "fit_tractable_gmm",
]

__version__ = "1.0.0.dev0"

"""Local Petrin state for one sequential orthogonal-projection update."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ...moments import AggregateIVMomentBuilder
from ...operators import PetrinMicroJackknifeBuilder
from .model import PetrinApplicationModel


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PetrinLocalState:
    """All local objects required for a Petrin SOP correction."""

    theta: FloatArray
    market_ids: NDArray
    tractable_moments: FloatArray
    demanding_moments: FloatArray
    tractable_jacobian: FloatArray
    demanding_jacobian: FloatArray
    aggregate_elapsed_seconds: float
    micro_elapsed_seconds: float

    def __post_init__(self) -> None:
        theta = np.asarray(self.theta, dtype=float).reshape(-1)
        market_ids = np.asarray(self.market_ids)
        g = np.asarray(self.tractable_moments, dtype=float)
        h = np.asarray(self.demanding_moments, dtype=float)
        G = np.asarray(self.tractable_jacobian, dtype=float)
        H = np.asarray(self.demanding_jacobian, dtype=float)

        if market_ids.ndim != 1:
            raise ValueError("market_ids must be one-dimensional.")
        if np.unique(market_ids).size != market_ids.size:
            raise ValueError("market_ids must be unique.")
        if g.ndim != 2 or h.ndim != 2:
            raise ValueError(
                "tractable_moments and demanding_moments must be matrices."
            )
        if g.shape[0] != market_ids.size:
            raise ValueError(
                "tractable_moments must have one row per market."
            )
        if h.shape[0] != market_ids.size:
            raise ValueError(
                "demanding_moments must have one row per market."
            )
        if G.shape != (g.shape[1], theta.size):
            raise ValueError(
                "tractable_jacobian has an incompatible shape."
            )
        if H.shape != (h.shape[1], theta.size):
            raise ValueError(
                "demanding_jacobian has an incompatible shape."
            )

        arrays = (theta, g, h, G, H)
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("PetrinLocalState contains non-finite values.")

        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "market_ids", market_ids)
        object.__setattr__(self, "tractable_moments", g)
        object.__setattr__(self, "demanding_moments", h)
        object.__setattr__(self, "tractable_jacobian", G)
        object.__setattr__(self, "demanding_jacobian", H)

    @property
    def n_markets(self) -> int:
        return int(self.market_ids.size)

    @property
    def n_tractable_moments(self) -> int:
        return int(self.tractable_moments.shape[1])

    @property
    def n_demanding_moments(self) -> int:
        return int(self.demanding_moments.shape[1])

    @property
    def parameter_dimension(self) -> int:
        return int(self.theta.size)

    @property
    def total_elapsed_seconds(self) -> float:
        return float(
            self.aggregate_elapsed_seconds + self.micro_elapsed_seconds
        )


class PetrinLocalStateBuilder:
    """Construct local aggregate and micro objects at a fixed parameter vector.

    Notes
    -----
    The current implementation performs one aggregate fixed evaluation and
    delegates the delete-one-market micro construction to
    ``PetrinMicroJackknifeBuilder``. It then performs one additional full
    micro evaluation to obtain PyBLP's public analytical micro Jacobian.
    This duplicate full micro evaluation is intentional in the first
    validated implementation and can later be removed by returning the full
    PyBLP result from the jackknife builder.
    """

    def __init__(
        self,
        model: PetrinApplicationModel,
        *,
        micro_builder: PetrinMicroJackknifeBuilder | None = None,
    ) -> None:
        self.model = model
        self.micro_builder = (
            micro_builder
            or PetrinMicroJackknifeBuilder(
                solver=model.solver,
            )
        )
        self.aggregate_builder = AggregateIVMomentBuilder()

    @staticmethod
    def _active_jacobian(
        matrix: Any,
        *,
        rows: int,
        parameters: int,
        name: str,
    ) -> FloatArray:
        value = np.asarray(matrix, dtype=float)

        if value.ndim != 2:
            raise ValueError(f"{name} must be two-dimensional.")
        if value.shape[0] < rows:
            raise ValueError(
                f"{name} has too few rows: expected at least {rows}."
            )
        if value.shape[1] < parameters:
            raise ValueError(
                f"{name} has too few columns: expected at least "
                f"{parameters}."
            )

        # PyBLP orders the active nonlinear parameter derivatives first.
        return value[:rows, :parameters]

    def build(self, theta: FloatArray) -> PetrinLocalState:
        theta_array = np.asarray(theta, dtype=float).reshape(-1)
        sigma, pi = self.model.structural_parameters(theta_array)

        aggregate_evaluation = self.model.evaluate_aggregate(theta_array)
        aggregate = aggregate_evaluation.moments
        aggregate_results = aggregate_evaluation.pyblp.results

        p = theta_array.size
        qg = aggregate.combined.shape[1]

        G = self._active_jacobian(
            aggregate_results.moments_jacobian,
            rows=qg,
            parameters=p,
            name="moments_jacobian",
        )

        micro = self.micro_builder.build(
            self.model.setup,
            sigma=sigma,
            pi=pi,
        )

        # One public full-sample micro solve supplies the analytical
        # derivative of the ten micro residuals with respect to active theta.
        full_micro_evaluation = self.model.solver.solve(
            self.model.setup,
            fidelity=self.micro_builder.fidelity_factory(
                self.model.setup
            ),
            sigma=sigma,
            pi=pi,
            include_micro=True,
            fixed_parameters=True,
            method="1s",
        )
        full_micro_results = full_micro_evaluation.results

        if not hasattr(
            full_micro_results,
            "micro_by_theta_jacobian",
        ):
            raise AttributeError(
                "PyBLP results do not expose "
                "'micro_by_theta_jacobian'."
            )

        H = np.asarray(
            full_micro_results.micro_by_theta_jacobian,
            dtype=float,
        )
        if H.shape != (
            micro.n_micro_moments,
            p,
        ):
            raise ValueError(
                "micro_by_theta_jacobian has an incompatible shape: "
                f"expected {(micro.n_micro_moments, p)}, got {H.shape}."
            )

        if not np.array_equal(
            aggregate.market_ids,
            micro.market_ids,
        ):
            raise ValueError(
                "Aggregate and micro market identifiers are not aligned."
            )

        return PetrinLocalState(
            theta=theta_array,
            market_ids=aggregate.market_ids,
            tractable_moments=aggregate.combined,
            demanding_moments=micro.contributions,
            tractable_jacobian=G,
            demanding_jacobian=H,
            aggregate_elapsed_seconds=(
                aggregate_evaluation.pyblp.elapsed_seconds
            ),
            micro_elapsed_seconds=(
                micro.total_elapsed_seconds
                + full_micro_evaluation.elapsed_seconds
            ),
        )

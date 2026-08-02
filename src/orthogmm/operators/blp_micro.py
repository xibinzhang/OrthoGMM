"""Market-level jackknife representation of PyBLP micro moments.

The public PyBLP API exposes the full-sample micro residual vector but not
market-level micro contributions. For the Petrin application, this module
constructs delete-one-market pseudo-values at fixed structural parameters.

For market t and T markets,

    p_t = T r_full - (T - 1) r_{(-t)},

where r_full is the full-sample micro residual vector and r_{(-t)} is the
residual vector after deleting market t. Because nonlinear jackknife
pseudo-values need not average exactly to the original statistic in finite
samples, the returned contributions are recentered:

    h_t = p_t + r_full - mean_s(p_s).

Thus mean_t(h_t) = r_full exactly, while cross-market variation is inherited
from the delete-one-market calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray

from ..blp import FidelityConfig
from ..model.petrin import PetrinProblem, build_petrin_problem
from ..solvers import PyBLPEvaluation, PyBLPSolver


FloatArray = NDArray[np.float64]


class PetrinSetupBuilder(Protocol):
    def __call__(
        self,
        *,
        exclude_market_ids: set[int] | None = None,
    ) -> PetrinProblem:
        ...


@dataclass(frozen=True, slots=True)
class MicroJackknifeResult:
    """Delete-one-market representation of PyBLP micro residuals."""

    market_ids: NDArray
    contributions: FloatArray
    raw_pseudo_values: FloatArray
    full_residual: FloatArray
    leave_one_out_residuals: FloatArray
    centering_adjustment: FloatArray
    full_elapsed_seconds: float
    leave_one_out_elapsed_seconds: FloatArray

    def __post_init__(self) -> None:
        market_ids = np.asarray(self.market_ids)
        contributions = np.asarray(self.contributions, dtype=float)
        raw = np.asarray(self.raw_pseudo_values, dtype=float)
        full = np.asarray(self.full_residual, dtype=float).reshape(-1)
        loo = np.asarray(self.leave_one_out_residuals, dtype=float)
        adjustment = np.asarray(
            self.centering_adjustment,
            dtype=float,
        ).reshape(-1)
        times = np.asarray(
            self.leave_one_out_elapsed_seconds,
            dtype=float,
        ).reshape(-1)

        if market_ids.ndim != 1:
            raise ValueError("market_ids must be one-dimensional.")

        T = market_ids.size
        q = full.size

        expected = (T, q)
        for name, value in (
            ("contributions", contributions),
            ("raw_pseudo_values", raw),
            ("leave_one_out_residuals", loo),
        ):
            if value.shape != expected:
                raise ValueError(
                    f"{name} must have shape {expected}; "
                    f"got {value.shape}."
                )

        if adjustment.shape != (q,):
            raise ValueError(
                f"centering_adjustment must have shape ({q},)."
            )

        if times.shape != (T,):
            raise ValueError(
                f"leave_one_out_elapsed_seconds must have shape ({T},)."
            )

        arrays = (
            contributions,
            raw,
            full,
            loo,
            adjustment,
            times,
        )
        if not all(np.all(np.isfinite(x)) for x in arrays):
            raise ValueError(
                "MicroJackknifeResult contains non-finite values."
            )

        if not np.allclose(
            contributions.mean(axis=0),
            full,
            rtol=1e-9,
            atol=1e-11,
        ):
            raise ValueError(
                "Centered contributions do not average to the "
                "full-sample micro residual."
            )

        object.__setattr__(self, "market_ids", market_ids)
        object.__setattr__(self, "contributions", contributions)
        object.__setattr__(self, "raw_pseudo_values", raw)
        object.__setattr__(self, "full_residual", full)
        object.__setattr__(self, "leave_one_out_residuals", loo)
        object.__setattr__(self, "centering_adjustment", adjustment)
        object.__setattr__(
            self,
            "leave_one_out_elapsed_seconds",
            times,
        )

    @property
    def n_markets(self) -> int:
        return int(self.market_ids.size)

    @property
    def n_micro_moments(self) -> int:
        return int(self.full_residual.size)

    @property
    def total_elapsed_seconds(self) -> float:
        return float(
            self.full_elapsed_seconds
            + self.leave_one_out_elapsed_seconds.sum()
        )


def centered_jackknife_pseudo_values(
    full_residual: FloatArray,
    leave_one_out_residuals: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Construct and recenter delete-one pseudo-values.

    Returns
    -------
    contributions
        Recentered market-level contributions whose mean exactly equals the
        full-sample residual vector.
    raw_pseudo_values
        Standard delete-one jackknife pseudo-values.
    centering_adjustment
        Common row adjustment applied to the raw pseudo-values.
    """

    full = np.asarray(full_residual, dtype=float).reshape(-1)
    loo = np.asarray(
        leave_one_out_residuals,
        dtype=float,
    )

    if loo.ndim != 2:
        raise ValueError(
            "leave_one_out_residuals must be a matrix."
        )

    T, q = loo.shape

    if T < 2:
        raise ValueError(
            "At least two markets are required for jackknife "
            "pseudo-values."
        )

    if full.shape != (q,):
        raise ValueError(
            "full_residual and leave_one_out_residuals have "
            "incompatible dimensions."
        )

    if not np.all(np.isfinite(full)) or not np.all(np.isfinite(loo)):
        raise ValueError("Micro residuals contain non-finite values.")

    raw = T * full[None, :] - (T - 1) * loo
    adjustment = full - raw.mean(axis=0)
    contributions = raw + adjustment[None, :]

    return contributions, raw, adjustment


class PetrinMicroJackknifeBuilder:
    """Construct the full market-by-micro-moment matrix for Petrin."""

    def __init__(
        self,
        *,
        solver: PyBLPSolver | None = None,
        setup_builder: PetrinSetupBuilder = build_petrin_problem,
        fidelity_factory: Callable[
            [PetrinProblem],
            FidelityConfig,
        ]
        | None = None,
    ) -> None:
        self.solver = solver or PyBLPSolver()
        self.setup_builder = setup_builder
        self.fidelity_factory = (
            fidelity_factory or self._default_fidelity
        )

    @staticmethod
    def _default_fidelity(
        setup: PetrinProblem,
    ) -> FidelityConfig:
        return FidelityConfig(
            name="petrin_micro_fixed",
            draws=setup.n_agents,
            contraction_tolerance=1e-10,
            max_iterations=1000,
            seed=0,
        )

    @staticmethod
    def _micro_residual(results: object) -> FloatArray:
        if not hasattr(results, "micro"):
            raise AttributeError(
                "PyBLP results do not expose 'micro'."
            )

        residual = np.asarray(
            getattr(results, "micro"),
            dtype=float,
        ).reshape(-1)

        if residual.size == 0:
            raise ValueError(
                "PyBLP evaluation returned no micro moments."
            )
        if not np.all(np.isfinite(residual)):
            raise ValueError(
                "PyBLP micro residuals contain non-finite values."
            )

        return residual

    def _solve(
        self,
        setup: PetrinProblem,
        *,
        sigma: FloatArray,
        pi: FloatArray,
    ) -> PyBLPEvaluation:
        return self.solver.solve(
            setup,
            fidelity=self.fidelity_factory(setup),
            sigma=sigma,
            pi=pi,
            include_micro=True,
            fixed_parameters=True,
            method="1s",
        )

    def build(
        self,
        setup: PetrinProblem,
        *,
        sigma: FloatArray,
        pi: FloatArray,
    ) -> MicroJackknifeResult:
        """Run one full and T leave-one-market-out fixed evaluations."""

        market_ids = np.asarray(setup.market_ids)
        if market_ids.ndim != 1 or market_ids.size < 2:
            raise ValueError(
                "setup.market_ids must contain at least two markets."
            )

        full_evaluation = self._solve(
            setup,
            sigma=sigma,
            pi=pi,
        )
        full_residual = self._micro_residual(
            full_evaluation.results
        )

        T = market_ids.size
        q = full_residual.size
        loo_residuals = np.empty((T, q), dtype=float)
        loo_times = np.empty(T, dtype=float)

        for row, market_id in enumerate(market_ids):
            reduced_setup = self.setup_builder(
                exclude_market_ids={int(market_id)},
            )

            start = perf_counter()
            reduced_evaluation = self._solve(
                reduced_setup,
                sigma=sigma,
                pi=pi,
            )
            wall_time = perf_counter() - start

            reduced_residual = self._micro_residual(
                reduced_evaluation.results
            )
            if reduced_residual.shape != (q,):
                raise ValueError(
                    "Leave-one-market-out micro residual dimension "
                    "does not match the full-sample dimension."
                )

            loo_residuals[row] = reduced_residual
            loo_times[row] = max(
                wall_time,
                reduced_evaluation.elapsed_seconds,
            )

        contributions, raw, adjustment = (
            centered_jackknife_pseudo_values(
                full_residual,
                loo_residuals,
            )
        )

        return MicroJackknifeResult(
            market_ids=market_ids,
            contributions=contributions,
            raw_pseudo_values=raw,
            full_residual=full_residual,
            leave_one_out_residuals=loo_residuals,
            centering_adjustment=adjustment,
            full_elapsed_seconds=(
                full_evaluation.elapsed_seconds
            ),
            leave_one_out_elapsed_seconds=loo_times,
        )

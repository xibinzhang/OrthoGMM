"""Petrin application model for OrthoGMM.

This module is the application boundary between the generic OrthoGMM engine
and the PyBLP Petrin benchmark. The first implementation supports the exact
market-level aggregate IV block. Market-level micro-moment contributions are
deliberately not fabricated and will be added in the next milestone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ...blp import FidelityConfig
from ...model.petrin import PetrinProblem, build_petrin_problem
from ...moments import AggregateIVMomentBuilder, AggregateIVMoments
from ...solvers import PyBLPEvaluation, PyBLPSolver


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ActiveParameterMap:
    """Map active Petrin nonlinear parameters to sigma and pi arrays."""

    sigma_shape: tuple[int, int]
    pi_shape: tuple[int, int]
    sigma_indices: NDArray[np.int_]
    pi_indices: NDArray[np.int_]

    @classmethod
    def from_setup(cls, setup: PetrinProblem) -> "ActiveParameterMap":
        sigma = np.asarray(setup.initial_sigma, dtype=float)
        pi = np.asarray(setup.initial_pi, dtype=float)

        return cls(
            sigma_shape=sigma.shape,
            pi_shape=pi.shape,
            sigma_indices=np.flatnonzero(sigma.reshape(-1) != 0),
            pi_indices=np.flatnonzero(pi.reshape(-1) != 0),
        )

    @property
    def dimension(self) -> int:
        return int(self.sigma_indices.size + self.pi_indices.size)

    def pack(
        self,
        sigma: FloatArray,
        pi: FloatArray,
    ) -> FloatArray:
        sigma_array = np.asarray(sigma, dtype=float)
        pi_array = np.asarray(pi, dtype=float)

        if sigma_array.shape != self.sigma_shape:
            raise ValueError("sigma has an incompatible shape.")
        if pi_array.shape != self.pi_shape:
            raise ValueError("pi has an incompatible shape.")

        return np.r_[
            sigma_array.reshape(-1)[self.sigma_indices],
            pi_array.reshape(-1)[self.pi_indices],
        ]

    def unpack(
        self,
        theta: FloatArray,
        *,
        sigma_template: FloatArray,
        pi_template: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        theta_array = np.asarray(theta, dtype=float).reshape(-1)

        if theta_array.size != self.dimension:
            raise ValueError(
                f"theta must have length {self.dimension}; "
                f"got {theta_array.size}."
            )
        if not np.all(np.isfinite(theta_array)):
            raise ValueError("theta contains non-finite values.")

        sigma = np.asarray(sigma_template, dtype=float).copy()
        pi = np.asarray(pi_template, dtype=float).copy()

        if sigma.shape != self.sigma_shape:
            raise ValueError("sigma_template has an incompatible shape.")
        if pi.shape != self.pi_shape:
            raise ValueError("pi_template has an incompatible shape.")

        split = self.sigma_indices.size
        sigma.reshape(-1)[self.sigma_indices] = theta_array[:split]
        pi.reshape(-1)[self.pi_indices] = theta_array[split:]

        return sigma, pi


@dataclass(frozen=True, slots=True)
class PetrinAggregateEvaluation:
    """One fixed-parameter aggregate evaluation and exact market moments."""

    pyblp: PyBLPEvaluation
    moments: AggregateIVMoments

    @property
    def theta(self) -> FloatArray:
        return np.asarray(
            self.pyblp.results.theta,
            dtype=float,
        ).reshape(-1)


class PetrinApplicationModel:
    """Application-specific Petrin model built on generic OrthoGMM components.

    Notes
    -----
    The tractable block is currently the exact 63-dimensional aggregate IV
    system. A demanding block based on market-level micro-moment
    pseudo-contributions will be added separately. Until then,
    ``demanding_moments`` raises ``NotImplementedError`` instead of returning
    an invalid placeholder.
    """

    def __init__(
        self,
        *,
        setup: PetrinProblem | None = None,
        solver: PyBLPSolver | None = None,
        aggregate_fidelity: FidelityConfig | None = None,
    ) -> None:
        self.setup = setup or build_petrin_problem()
        self.solver = solver or PyBLPSolver()
        self.aggregate_fidelity = (
            aggregate_fidelity
            or FidelityConfig(
                name="petrin_aggregate",
                draws=self.setup.n_agents,
                contraction_tolerance=1e-10,
                max_iterations=1000,
                seed=0,
            )
        )
        self.parameter_map = ActiveParameterMap.from_setup(self.setup)
        self._aggregate_builder = AggregateIVMomentBuilder()

    @property
    def theta0(self) -> FloatArray:
        """Active nonlinear Petrin starting values."""

        return self.parameter_map.pack(
            self.setup.initial_sigma,
            self.setup.initial_pi,
        )

    @property
    def parameter_dimension(self) -> int:
        return self.parameter_map.dimension

    def structural_parameters(
        self,
        theta: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        """Convert an OrthoGMM vector into PyBLP sigma and pi blocks."""

        return self.parameter_map.unpack(
            theta,
            sigma_template=self.setup.initial_sigma,
            pi_template=self.setup.initial_pi,
        )

    def evaluate_aggregate(
        self,
        theta: FloatArray,
    ) -> PetrinAggregateEvaluation:
        """Evaluate aggregate PyBLP moments once at fixed parameters."""

        sigma, pi = self.structural_parameters(theta)

        evaluation = self.solver.solve(
            self.setup,
            fidelity=self.aggregate_fidelity,
            sigma=sigma,
            pi=pi,
            include_micro=False,
            fixed_parameters=True,
            method="1s",
        )

        moments = self._aggregate_builder.from_pyblp(
            self.setup.problem,
            evaluation.results,
        )

        return PetrinAggregateEvaluation(
            pyblp=evaluation,
            moments=moments,
        )

    def tractable_moments(
        self,
        theta: FloatArray,
    ) -> FloatArray:
        """Return exact market-level aggregate IV moments."""

        return self.evaluate_aggregate(theta).moments.combined

    def demanding_moments(
        self,
        theta: FloatArray,
    ) -> FloatArray:
        """Return market-level micro corrections once implemented."""

        raise NotImplementedError(
            "Market-level micro-moment pseudo-contributions have not "
            "yet been implemented. The model intentionally refuses to "
            "fabricate a demanding block."
        )

    def reconstruct(self, theta: FloatArray) -> dict[str, Any]:
        """Return structural PyBLP blocks and one aggregate evaluation."""

        evaluation = self.evaluate_aggregate(theta)
        results = evaluation.pyblp.results

        return {
            "sigma": np.asarray(results.sigma, dtype=float),
            "pi": np.asarray(results.pi, dtype=float),
            "beta": np.asarray(results.beta, dtype=float),
            "gamma": np.asarray(results.gamma, dtype=float),
            "objective": float(np.asarray(results.objective).squeeze()),
            "elapsed_seconds": evaluation.pyblp.elapsed_seconds,
        }

"""Thin PyBLP solver wrapper for fixed and optimized Petrin evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from ..blp import FidelityConfig
from ..model.petrin import PetrinProblem


@dataclass(frozen=True, slots=True)
class PyBLPEvaluation:
    """Raw PyBLP results and computational diagnostics."""

    results: Any
    elapsed_seconds: float
    include_micro: bool
    fixed_parameters: bool
    fidelity: FidelityConfig


class PyBLPSolver:
    """Translate OrthoGMM fidelity settings into PyBLP solve options.

    The wrapper deliberately returns raw ``ProblemResults`` objects. It does
    not extract moments, construct projections, or perform OrthoGMM updates.
    """

    def __init__(
        self,
        *,
        optimization_method: str = "bfgs",
        gradient_tolerance: float = 1e-4,
        se_type: str = "clustered",
        weighting_type: str = "clustered",
    ) -> None:
        if gradient_tolerance <= 0:
            raise ValueError(
                "gradient_tolerance must be positive."
            )

        self.optimization_method = optimization_method
        self.gradient_tolerance = float(gradient_tolerance)
        self.se_type = se_type
        self.weighting_type = weighting_type

    def solve(
        self,
        setup: PetrinProblem,
        *,
        fidelity: FidelityConfig,
        sigma: np.ndarray | None = None,
        pi: np.ndarray | None = None,
        include_micro: bool = False,
        fixed_parameters: bool = False,
        weighting_matrix: np.ndarray | None = None,
        initial_update: bool | None = None,
        method: str = "1s",
        check_optimality: str | None = None,
    ) -> PyBLPEvaluation:
        """Run one PyBLP solve or fixed-parameter evaluation.

        Parameters
        ----------
        setup
            Petrin problem inputs.
        fidelity
            Numerical configuration. ``draws`` is recorded for diagnostics;
            the Petrin agent data already contain fixed simulation nodes, so
            this wrapper does not resample agents.
        sigma, pi
            Structural parameter blocks. Defaults are the Petrin starting
            values.
        include_micro
            Include the Petrin micro moments.
        fixed_parameters
            Use ``Optimization("return")`` so PyBLP evaluates at the supplied
            nonlinear parameters without nonlinear optimization.
        weighting_matrix
            Optional complete weighting matrix.
        initial_update
            Optional PyBLP initial-weight update flag.
        method
            PyBLP GMM method, usually ``"1s"`` or ``"2s"``.
        check_optimality
            Optional PyBLP optimality-check setting.
        """

        try:
            import pyblp
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "PyBLP is required for PyBLPSolver."
            ) from exc

        if method not in {"1s", "2s"}:
            raise ValueError("method must be '1s' or '2s'.")

        sigma_value = np.asarray(
            setup.initial_sigma if sigma is None else sigma,
            dtype=float,
        )
        pi_value = np.asarray(
            setup.initial_pi if pi is None else pi,
            dtype=float,
        )

        if sigma_value.shape != setup.initial_sigma.shape:
            raise ValueError(
                "sigma has an incompatible shape."
            )
        if pi_value.shape != setup.initial_pi.shape:
            raise ValueError(
                "pi has an incompatible shape."
            )
        if not np.all(np.isfinite(sigma_value)):
            raise ValueError("sigma contains non-finite values.")
        if not np.all(np.isfinite(pi_value)):
            raise ValueError("pi contains non-finite values.")

        iteration_options: dict[str, Any] = {
            "atol": fidelity.contraction_tolerance,
        }
        if fidelity.max_iterations > 0:
            iteration_options["max_evaluations"] = (
                fidelity.max_iterations
            )

        optimization = (
            pyblp.Optimization("return")
            if fixed_parameters
            else pyblp.Optimization(
                self.optimization_method,
                {"gtol": self.gradient_tolerance},
            )
        )

        kwargs: dict[str, Any] = {
            "sigma": sigma_value,
            "pi": pi_value,
            "method": method,
            "optimization": optimization,
            "iteration": pyblp.Iteration(
                "squarem",
                iteration_options,
            ),
            "se_type": self.se_type,
            "W_type": self.weighting_type,
            "micro_moments": (
                setup.micro_moments if include_micro else ()
            ),
        }

        if check_optimality is not None:
            kwargs["check_optimality"] = check_optimality
        elif fixed_parameters:
            kwargs["check_optimality"] = "gradient"

        if weighting_matrix is not None:
            W = np.asarray(weighting_matrix, dtype=float)
            if W.ndim != 2 or W.shape[0] != W.shape[1]:
                raise ValueError(
                    "weighting_matrix must be square."
                )
            if not np.all(np.isfinite(W)):
                raise ValueError(
                    "weighting_matrix contains non-finite values."
                )
            kwargs["W"] = W

        if initial_update is not None:
            kwargs["initial_update"] = bool(initial_update)

        # Preserve room for future fidelity-specific PyBLP options without
        # allowing protected arguments to be overwritten.
        protected = set(kwargs)
        overlap = protected.intersection(fidelity.options)
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(
                "fidelity.options must not override solver-controlled "
                f"keys: {names}."
            )
        kwargs.update(fidelity.options)

        start = perf_counter()
        results = setup.problem.solve(**kwargs)
        elapsed = perf_counter() - start

        return PyBLPEvaluation(
            results=results,
            elapsed_seconds=elapsed,
            include_micro=include_micro,
            fixed_parameters=fixed_parameters,
            fidelity=fidelity,
        )

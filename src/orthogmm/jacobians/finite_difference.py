from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..differentiation import finite_difference_jacobian
from ..exceptions import ModelContractError
from ..types import Array
from .base import JacobianContext, JacobianStrategy


class FiniteDifferenceJacobian(JacobianStrategy):
    """Construct a Jacobian from finite differences of mean moments."""

    def __init__(self, *, rel_step: float | None = None) -> None:
        self.rel_step = rel_step

    def compute(self, context: JacobianContext) -> Array:
        moment_name = (
            "tractable_moments"
            if context.moment_kind == "tractable"
            else "demanding_moments"
        )

        moment_method = getattr(context.model, moment_name, None)

        if not callable(moment_method):
            raise ModelContractError(
                f"The model does not provide {moment_name}()."
            )

        def mean_function(theta: Array) -> Array:
            values = np.asarray(moment_method(theta), dtype=float)

            if values.ndim != 2:
                raise ModelContractError(
                    f"{moment_name} must return a "
                    "unit-by-moment matrix."
                )

            if context.moment_kind == "tractable":
                context.counts.tractable_moments += 1
            else:
                context.counts.demanding_moments_derivative += 1

            return values.mean(axis=0)

        jacobian = finite_difference_jacobian(
            mean_function,
            np.asarray(context.theta, dtype=float),
            rel_step=self.rel_step,
            bounds=context.bounds,
        )

        self._validate(jacobian, context=context)
        return jacobian

    @staticmethod
    def _validate(
        jacobian: Array,
        *,
        context: JacobianContext,
    ) -> None:
        if jacobian.ndim != 2 or jacobian.shape[1] != context.theta.size:
            raise ModelContractError(
                "Finite-difference Jacobian must return a "
                "moment-by-parameter matrix with "
                f"{context.theta.size} columns."
            )

        if not np.all(np.isfinite(jacobian)):
            raise ModelContractError(
                "Finite-difference Jacobian contains "
                "non-finite values."
            )

from __future__ import annotations

import numpy as np

from ..exceptions import ModelContractError
from ..types import Array
from .base import JacobianContext, JacobianStrategy


class AnalyticalJacobian(JacobianStrategy):
    """Construct a Jacobian from a model-supplied analytical method."""

    def compute(self, context: JacobianContext) -> Array:
        if context.moment_kind == "tractable":
            method_name = "tractable_jacobian"
        else:
            method_name = "demanding_jacobian"

        method = getattr(context.model, method_name, None)

        if not callable(method):
            raise NotImplementedError(
                f"The model does not provide {method_name}()."
            )

        try:
            jacobian = np.asarray(
                method(context.theta),
                dtype=float,
            )
        except NotImplementedError:
            raise

        if context.moment_kind == "tractable":
            context.counts.tractable_jacobian += 1
        else:
            context.counts.demanding_jacobian += 1

        self._validate(
            jacobian,
            theta=context.theta,
            method_name=method_name,
        )

        return jacobian

    @staticmethod
    def _validate(
        jacobian: Array,
        *,
        theta: Array,
        method_name: str,
    ) -> None:
        if jacobian.ndim != 2 or jacobian.shape[1] != theta.size:
            raise ModelContractError(
                f"{method_name} must return a "
                f"moment-by-parameter matrix with "
                f"{theta.size} columns."
            )

        if not np.all(np.isfinite(jacobian)):
            raise ModelContractError(
                f"{method_name} returned non-finite values."
            )

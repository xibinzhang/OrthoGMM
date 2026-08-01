from __future__ import annotations

from ..types import Array
from .analytical import AnalyticalJacobian
from .base import JacobianContext, JacobianStrategy
from .finite_difference import FiniteDifferenceJacobian


class FallbackJacobian(JacobianStrategy):
    """Use an analytical Jacobian when available.

    If the model does not implement the requested analytical Jacobian,
    automatically fall back to finite differences of the corresponding
    mean moment function.
    """

    def __init__(self, *, rel_step: float | None = None) -> None:
        self.analytical = AnalyticalJacobian()
        self.finite_difference = FiniteDifferenceJacobian(
            rel_step=rel_step,
        )

    def compute(self, context: JacobianContext) -> Array:
        try:
            return self.analytical.compute(context)
        except NotImplementedError:
            return self.finite_difference.compute(context)
        
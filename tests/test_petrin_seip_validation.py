from types import SimpleNamespace

import numpy as np

from orthogmm.applications.petrin.validation import PetrinSEIPValidator


class FakeSolver:
    def __init__(self) -> None:
        self.calls = []

    def solve(self, *args, **kwargs):
        self.calls.append(kwargs)
        sigma = np.asarray(kwargs["sigma"], dtype=float)
        pi = np.asarray(kwargs["pi"], dtype=float)
        theta_sum = float(sigma.sum() + pi.sum())
        results = SimpleNamespace(
            objective=np.array([theta_sum**2]),
            projected_gradient_norm=np.array([abs(theta_sum)]),
            fixed_point_iterations=np.array([[2, 3]]),
            contraction_evaluations=np.array([[5, 7]]),
        )
        return SimpleNamespace(
            results=results,
            elapsed_seconds=0.25,
        )


class FakeModel:
    parameter_dimension = 2
    setup = object()
    high_fidelity = object()

    def __init__(self) -> None:
        self.solver = FakeSolver()

    def structural_parameters(self, theta):
        return (
            np.array([[theta[0]]]),
            np.array([[theta[1]]]),
        )


def test_validator_uses_fixed_parameter_solves() -> None:
    model = FakeModel()
    validator = PetrinSEIPValidator(model)

    result = validator.compare(
        np.array([1.0, 1.0]),
        np.array([0.5, 0.5]),
    )

    assert len(model.solver.calls) == 2
    assert all(
        call["fixed_parameters"] is True
        for call in model.solver.calls
    )
    assert all(
        call["include_micro"] is True
        for call in model.solver.calls
    )
    assert result.objective_improved
    assert result.localized.fixed_point_iterations == 5
    assert result.localized.contraction_evaluations == 12

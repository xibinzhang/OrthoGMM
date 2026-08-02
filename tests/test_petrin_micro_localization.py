from types import SimpleNamespace

import numpy as np

from orthogmm import FidelityConfig
from orthogmm.applications.petrin.micro_localization import (
    PetrinMicroLocalizer,
)


class FakeSolver:
    def __init__(self) -> None:
        self.kwargs = None

    def solve(self, *args, **kwargs):
        self.kwargs = kwargs
        results = SimpleNamespace(
            theta=np.array([1.2, -0.3]),
            objective=np.array([5.0]),
            projected_gradient_norm=np.array([0.01]),
            optimization_converged=True,
            optimization_iterations=np.array([4]),
            objective_evaluations=np.array([7]),
            fixed_point_iterations=np.array([20]),
            contraction_evaluations=np.array([60]),
        )
        return SimpleNamespace(
            results=results,
            elapsed_seconds=2.0,
        )


class FakeSetup:
    n_agents = 100


class FakeModel:
    theta0 = np.array([1.0, -0.5])
    parameter_dimension = 2
    setup = FakeSetup()

    def __init__(self) -> None:
        self.solver = FakeSolver()

    def structural_parameters(self, theta):
        return (
            np.array([[theta[0]]]),
            np.array([[theta[1]]]),
        )


def test_micro_localizer_includes_micro_moments() -> None:
    model = FakeModel()
    fidelity = FidelityConfig(
        name="coarse",
        draws=100,
        contraction_tolerance=1e-5,
        max_iterations=50,
    )

    result = PetrinMicroLocalizer(
        model,
        fidelity=fidelity,
    ).fit()

    assert model.solver.kwargs["include_micro"] is True
    assert model.solver.kwargs["fixed_parameters"] is False
    assert model.solver.kwargs["fidelity"] is fidelity
    np.testing.assert_allclose(
        result.theta_localized,
        np.array([1.2, -0.3]),
    )
    assert result.converged

from types import SimpleNamespace

import numpy as np

from orthogmm.applications.petrin.localization import (
    PetrinLocalizationResult,
    PetrinTractableLocalizer,
)


def test_localization_result_update_diagnostics() -> None:
    result = PetrinLocalizationResult(
        theta_initial=np.array([1.0, 2.0]),
        theta_localized=np.array([1.1, 1.8]),
        objective=3.0,
        projected_gradient_norm=0.1,
        converged=True,
        optimization_iterations=4,
        objective_evaluations=8,
        fixed_point_iterations=20,
        contraction_evaluations=60,
        elapsed_seconds=2.5,
        lower_bound_hits=(),
        upper_bound_hits=(1,),
        pyblp_results=object(),
    )

    np.testing.assert_allclose(
        result.update,
        np.array([0.1, -0.2]),
    )
    assert result.update_norm > 0
    assert result.on_boundary


class FakeMap:
    sigma_indices = np.array([0, 3])
    pi_indices = np.array([1])


class FakeSetup:
    sigma_bounds = (
        np.array([[0.0, 0.0], [0.0, 0.0]]),
        np.array([[10.0, 0.0], [0.0, 10.0]]),
    )
    pi_bounds = (
        np.array([[0.0, -np.inf]]),
        np.array([[0.0, np.inf]]),
    )


class FakeSolver:
    def solve(self, *args, **kwargs):
        results = SimpleNamespace(
            theta=np.array([1.5, 2.5, -0.4]),
            objective=np.array([7.0]),
            projected_gradient_norm=np.array([0.02]),
            optimization_converged=True,
            optimization_iterations=np.array([3]),
            objective_evaluations=np.array([5]),
            fixed_point_iterations=np.array([12]),
            contraction_evaluations=np.array([36]),
        )
        return SimpleNamespace(
            results=results,
            elapsed_seconds=1.2,
        )


class FakeModel:
    theta0 = np.array([1.0, 2.0, -0.5])
    parameter_dimension = 3
    parameter_map = FakeMap()
    setup = FakeSetup()
    aggregate_fidelity = object()
    solver = FakeSolver()

    def structural_parameters(self, theta):
        sigma = np.array([[theta[0], 0.0], [0.0, theta[1]]])
        pi = np.array([[0.0, theta[2]]])
        return sigma, pi


def test_localizer_delegates_aggregate_optimization() -> None:
    result = PetrinTractableLocalizer(FakeModel()).fit()

    np.testing.assert_allclose(
        result.theta_localized,
        np.array([1.5, 2.5, -0.4]),
    )
    assert result.converged
    assert result.objective == 7.0
    assert result.optimization_iterations == 3

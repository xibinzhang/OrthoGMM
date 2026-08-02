import numpy as np

from orthogmm.optimization.trust_region import QuadraticTrustRegion


def test_unconstrained_solution_is_returned_when_feasible() -> None:
    J = np.diag([2.0, 4.0])
    s = np.array([-2.0, 4.0])

    result = QuadraticTrustRegion(
        radius=10.0,
        metric_type="euclidean",
    ).solve(J, s)

    np.testing.assert_allclose(
        result.step,
        np.array([1.0, -1.0]),
        atol=1e-10,
    )
    assert not result.active
    assert result.lagrange_multiplier == 0.0


def test_active_euclidean_solution_hits_radius() -> None:
    J = np.eye(2)
    s = np.array([-3.0, -4.0])

    result = QuadraticTrustRegion(
        radius=1.0,
        metric_type="euclidean",
    ).solve(J, s)

    np.testing.assert_allclose(
        result.step,
        np.array([0.6, 0.8]),
        atol=1e-8,
    )
    assert result.active
    assert abs(result.metric_norm - 1.0) < 1e-8
    assert result.lagrange_multiplier > 0.0


def test_parameter_scale_metric_uses_relative_distance() -> None:
    J = np.eye(2)
    s = np.array([-10.0, -10.0])
    theta = np.array([1.0, 100.0])

    result = QuadraticTrustRegion(
        radius=0.1,
        metric_type="parameter_scale",
    ).solve(J, s, theta=theta)

    relative_norm = np.sqrt(
        result.step[0] ** 2
        + (result.step[1] / 100.0) ** 2
    )
    assert abs(relative_norm - 0.1) < 1e-8
    assert abs(result.step[1]) > abs(result.step[0])


def test_information_metric_hits_statistical_radius() -> None:
    J = np.diag([1.0, 9.0])
    s = np.array([-2.0, -2.0])

    result = QuadraticTrustRegion(
        radius=0.5,
        metric_type="information",
    ).solve(J, s)

    statistical_norm = np.sqrt(result.step @ J @ result.step)
    assert abs(statistical_norm - 0.5) < 1e-8
    assert result.active

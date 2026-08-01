import numpy as np

from orthogmm import (
    RandomCoefficientIntegration,
    RandomCoefficientMomentModel,
    fit_seip,
    fit_tractable_gmm,
)
from orthogmm.simulation import NonlinearRandomCoefficientDesign


def make_data(n: int = 200, seed: int = 123):
    return NonlinearRandomCoefficientDesign(
        n=n,
        alpha=0.0,
        beta=1.0,
        lambda_=0.5,
        sigma=0.8,
        seed=seed,
    ).generate()


def test_random_coefficient_tractable_moments() -> None:
    data = make_data()
    model = RandomCoefficientMomentModel.from_data(data)
    theta = np.array([0.2, 0.9])

    residual = data["y"] - theta[0] - theta[1] * data["x"]
    expected = np.column_stack(
        (residual, data["x"] * residual)
    )

    np.testing.assert_allclose(
        model.tractable_moments(theta),
        expected,
    )


def test_random_coefficient_tractable_jacobian() -> None:
    data = make_data()
    model = RandomCoefficientMomentModel.from_data(data)
    theta = np.array([0.0, 1.0])

    expected = np.array(
        [
            [-1.0, -np.mean(data["x"])],
            [-np.mean(data["x"]), -np.mean(data["x"] ** 2)],
        ]
    )

    np.testing.assert_allclose(
        model.tractable_jacobian(theta),
        expected,
    )


def test_random_coefficient_demanding_moment_shape() -> None:
    data = make_data()
    model = RandomCoefficientMomentModel.from_data(
        data,
        basis_k=3,
        basis_family="fourier",
    )

    moments = model.demanding_moments(
        np.array([0.0, 1.0])
    )

    assert moments.shape == (200, 6)
    assert np.all(np.isfinite(moments))


def test_quadrature_demanding_moments_are_reproducible() -> None:
    data = make_data()
    integration = RandomCoefficientIntegration(
        mode="quadrature",
        q_nodes=20,
    )
    first = RandomCoefficientMomentModel.from_data(
        data,
        integration=integration,
    )
    second = RandomCoefficientMomentModel.from_data(
        data,
        integration=integration,
    )
    theta = np.array([0.0, 1.0])

    np.testing.assert_allclose(
        first.demanding_moments(theta),
        second.demanding_moments(theta),
    )


def test_simulation_demanding_moments_are_reproducible() -> None:
    data = make_data()
    integration = RandomCoefficientIntegration(
        mode="simulation",
        simulation_draws=100,
        seed=789,
    )
    first = RandomCoefficientMomentModel.from_data(
        data,
        integration=integration,
    )
    second = RandomCoefficientMomentModel.from_data(
        data,
        integration=integration,
    )
    theta = np.array([0.0, 1.0])

    np.testing.assert_allclose(
        first.demanding_moments(theta),
        second.demanding_moments(theta),
    )


def test_random_coefficient_model_supports_tractable_gmm() -> None:
    data = make_data(n=300)
    model = RandomCoefficientMomentModel.from_data(data)

    result = fit_tractable_gmm(
        model,
        theta0=np.array([0.0, 0.0]),
    )

    assert result.theta.shape == (2,)
    assert np.all(np.isfinite(result.theta))
    assert np.all(np.isfinite(result.standard_errors))


def test_random_coefficient_model_supports_seip() -> None:
    data = make_data(n=300)
    model = RandomCoefficientMomentModel.from_data(
        data,
        basis_k=1,
        basis_family="fourier",
        integration=RandomCoefficientIntegration(
            mode="quadrature",
            q_nodes=20,
        ),
    )

    preliminary = fit_tractable_gmm(
        model,
        theta0=np.array([0.0, 0.0]),
    ).theta

    result = fit_seip(
        model,
        theta0=np.array([0.0, 0.0]),
        preliminary_theta=preliminary,
        ridge=1e-8,
    )

    assert result.theta.shape == (2,)
    assert result.projection.shape == (2, 2)
    assert result.R.shape == (2, 2)
    assert np.all(np.isfinite(result.theta))
    assert result.counts.demanding_moments_total > 0

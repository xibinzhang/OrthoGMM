import numpy as np
import pytest

from orthogmm.simulation import NonlinearRandomCoefficientDesign


def test_nonlinear_random_coefficient_shapes() -> None:
    design = NonlinearRandomCoefficientDesign(
        n=200,
        alpha=0.0,
        beta=1.0,
        lambda_=0.5,
        sigma=0.8,
        seed=123,
    )
    data = design.generate()

    assert data["x"].shape == (200,)
    assert data["y"].shape == (200,)
    assert data["u"].shape == (200,)
    assert data["epsilon"].shape == (200,)
    assert data["random_coefficient"].shape == (200,)
    assert data["theta_true"].shape == (2,)


def test_nonlinear_random_coefficient_reproducibility() -> None:
    design = NonlinearRandomCoefficientDesign(n=100, seed=123)
    first = design.generate()
    second = design.generate()

    for name in (
        "x",
        "y",
        "u",
        "epsilon",
        "random_coefficient",
        "theta_true",
    ):
        np.testing.assert_allclose(first[name], second[name])


def test_nonlinear_random_coefficient_replication_seed() -> None:
    design = NonlinearRandomCoefficientDesign(n=100, seed=123)
    first = design.generate(seed=1)
    second = design.generate(seed=2)

    assert not np.array_equal(first["y"], second["y"])


def test_random_coefficient_is_centered_in_large_sample() -> None:
    design = NonlinearRandomCoefficientDesign(
        n=200_000,
        lambda_=0.5,
        seed=123,
    )
    data = design.generate()

    assert abs(np.mean(data["random_coefficient"])) < 0.01


def test_conditional_mean_parameters_are_recorded() -> None:
    design = NonlinearRandomCoefficientDesign(
        alpha=0.2,
        beta=1.3,
    )

    np.testing.assert_allclose(
        design.theta_true,
        np.array([0.2, 1.3]),
    )


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("n", 0),
        ("lambda_", -0.1),
        ("sigma", 0.0),
        ("sigma", np.inf),
    ],
)
def test_invalid_nonlinear_design_arguments(
    argument: str,
    value: object,
) -> None:
    with pytest.raises(ValueError):
        NonlinearRandomCoefficientDesign(**{argument: value})

import numpy as np
import pytest

from orthogmm.simulation import LinearIVDesign


def test_linear_iv_shapes() -> None:
    design = LinearIVDesign(
        n=200,
        beta=(1.0, -0.5),
        n_instruments=4,
        seed=123,
    )

    data = design.generate()

    assert data["y"].shape == (200,)
    assert data["X"].shape == (200, 2)
    assert data["Z"].shape == (200, 6)
    assert data["theta_true"].shape == (2,)


def test_linear_iv_reproducibility() -> None:
    design = LinearIVDesign(n=100, seed=123)

    first = design.generate()
    second = design.generate()

    np.testing.assert_allclose(first["y"], second["y"])
    np.testing.assert_allclose(first["X"], second["X"])
    np.testing.assert_allclose(first["Z"], second["Z"])


def test_replication_specific_seed() -> None:
    design = LinearIVDesign(n=100, seed=123)

    first = design.generate(seed=1)
    second = design.generate(seed=2)

    assert not np.array_equal(first["y"], second["y"])


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("n", 0),
        ("n_instruments", 0),
        ("instrument_strength", 0.0),
        ("sigma", 0.0),
        ("endogeneity", 1.0),
    ],
)
def test_invalid_design_arguments(argument: str, value: object) -> None:
    arguments = {argument: value}

    with pytest.raises(ValueError):
        LinearIVDesign(**arguments)

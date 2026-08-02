import numpy as np
import pytest

from orthogmm.applications.petrin import (
    ActiveParameterMap,
    PetrinApplicationModel,
)
from orthogmm.model.petrin import PetrinProblem


class FakeSetup:
    def __init__(self):
        self.initial_sigma = np.diag([1.0, 0.0, 2.0])
        self.initial_pi = np.array(
            [
                [0.0, 3.0],
                [0.0, 0.0],
                [4.0, 0.0],
            ]
        )
        self.n_agents = 10


def test_active_parameter_map_round_trip() -> None:
    setup = FakeSetup()
    mapping = ActiveParameterMap.from_setup(setup)

    theta = mapping.pack(
        setup.initial_sigma,
        setup.initial_pi,
    )
    sigma, pi = mapping.unpack(
        theta,
        sigma_template=setup.initial_sigma,
        pi_template=setup.initial_pi,
    )

    assert mapping.dimension == 4
    np.testing.assert_allclose(sigma, setup.initial_sigma)
    np.testing.assert_allclose(pi, setup.initial_pi)


def test_active_parameter_map_changes_only_active_entries() -> None:
    setup = FakeSetup()
    mapping = ActiveParameterMap.from_setup(setup)
    theta = mapping.pack(
        setup.initial_sigma,
        setup.initial_pi,
    )
    changed = theta + np.arange(theta.size)

    sigma, pi = mapping.unpack(
        changed,
        sigma_template=setup.initial_sigma,
        pi_template=setup.initial_pi,
    )

    assert sigma[0, 1] == 0.0
    assert sigma[1, 0] == 0.0
    assert pi[0, 0] == 0.0
    assert pi[1, 1] == 0.0


def test_demanding_block_is_not_fabricated() -> None:
    model = object.__new__(PetrinApplicationModel)

    with pytest.raises(NotImplementedError, match="micro-moment"):
        model.demanding_moments(np.zeros(1))

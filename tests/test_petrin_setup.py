import numpy as np
import pytest

pytest.importorskip("pyblp")

from orthogmm.model.petrin import build_petrin_problem


@pytest.fixture(scope="module")
def setup():
    return build_petrin_problem()


def test_petrin_problem_constructs(setup) -> None:
    assert setup.problem is not None
    assert setup.n_markets == 13
    assert setup.n_products == 2407
    assert setup.n_agents == 13000


def test_petrin_initial_parameter_shapes(setup) -> None:
    assert setup.initial_sigma.shape == (11, 11)
    assert setup.initial_pi.shape == (11, 9)
    assert np.all(np.isfinite(setup.initial_sigma))
    assert np.all(np.isfinite(setup.initial_pi))


def test_petrin_micro_moments(setup) -> None:
    assert len(setup.micro_moments) == 10
    names = [moment.name for moment in setup.micro_moments]
    assert "E[age_i | mi_j]" in names
    assert "E[1{j > 0} | high_i]" in names


def test_petrin_market_exclusion(setup) -> None:
    excluded = setup.market_ids[0]
    reduced = build_petrin_problem(
        exclude_market_ids={excluded},
    )
    assert reduced.n_markets == setup.n_markets - 1
    assert excluded not in set(reduced.market_ids)


def test_excluding_all_markets_is_rejected(setup) -> None:
    with pytest.raises(ValueError):
        build_petrin_problem(
            exclude_market_ids=set(setup.market_ids),
        )

import numpy as np
import pytest

from orthogmm.moments import AggregateIVMomentBuilder


def make_inputs():
    market_ids = np.array([2, 1, 2, 1, 3])
    ZD = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
        ]
    )
    xi = np.array([1.0, -1.0, 2.0, 0.5, -0.5])
    ZS = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
        ]
    )
    omega = np.array([0.5, 1.0, -1.0, 2.0, -0.5])
    return market_ids, ZD, xi, ZS, omega


def test_market_average_reproduces_product_average() -> None:
    market_ids, ZD, xi, ZS, omega = make_inputs()

    out = AggregateIVMomentBuilder().build(
        product_market_ids=market_ids,
        demand_instruments=ZD,
        demand_residuals=xi,
        supply_instruments=ZS,
        supply_residuals=omega,
    )

    expected = np.r_[
        ZD.T @ xi / xi.size,
        ZS.T @ omega / omega.size,
    ]

    np.testing.assert_allclose(out.average, expected)


def test_market_order_is_respected() -> None:
    market_ids, ZD, xi, ZS, omega = make_inputs()

    out = AggregateIVMomentBuilder().build(
        product_market_ids=market_ids,
        demand_instruments=ZD,
        demand_residuals=xi,
        supply_instruments=ZS,
        supply_residuals=omega,
        market_order=np.array([3, 1, 2]),
    )

    np.testing.assert_array_equal(
        out.market_ids,
        np.array([3, 1, 2]),
    )


def test_product_permutation_does_not_change_grouped_moments() -> None:
    market_ids, ZD, xi, ZS, omega = make_inputs()
    builder = AggregateIVMomentBuilder()

    original = builder.build(
        product_market_ids=market_ids,
        demand_instruments=ZD,
        demand_residuals=xi,
        supply_instruments=ZS,
        supply_residuals=omega,
    )

    permutation = np.array([4, 2, 0, 3, 1])
    shuffled = builder.build(
        product_market_ids=market_ids[permutation],
        demand_instruments=ZD[permutation],
        demand_residuals=xi[permutation],
        supply_instruments=ZS[permutation],
        supply_residuals=omega[permutation],
    )

    np.testing.assert_array_equal(
        original.market_ids,
        shuffled.market_ids,
    )
    np.testing.assert_allclose(
        original.combined,
        shuffled.combined,
    )


def test_supply_inputs_must_be_supplied_together() -> None:
    market_ids, ZD, xi, ZS, _ = make_inputs()

    with pytest.raises(ValueError, match="supplied together"):
        AggregateIVMomentBuilder().build(
            product_market_ids=market_ids,
            demand_instruments=ZD,
            demand_residuals=xi,
            supply_instruments=ZS,
        )

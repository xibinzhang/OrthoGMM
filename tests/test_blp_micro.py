import numpy as np
import pytest

from orthogmm.operators.blp_micro import (
    MicroJackknifeResult,
    centered_jackknife_pseudo_values,
)


def test_centered_pseudo_values_average_to_full_residual() -> None:
    full = np.array([1.0, -2.0])
    loo = np.array(
        [
            [0.8, -2.1],
            [1.1, -1.8],
            [0.9, -2.2],
        ]
    )

    contributions, raw, adjustment = (
        centered_jackknife_pseudo_values(full, loo)
    )

    expected_raw = 3 * full - 2 * loo
    np.testing.assert_allclose(raw, expected_raw)
    np.testing.assert_allclose(
        contributions.mean(axis=0),
        full,
    )
    np.testing.assert_allclose(
        contributions,
        raw + adjustment,
    )


def test_centered_pseudo_values_preserve_cross_market_differences() -> None:
    full = np.array([0.5])
    loo = np.array([[0.3], [0.4], [0.7]])

    contributions, raw, _ = (
        centered_jackknife_pseudo_values(full, loo)
    )

    np.testing.assert_allclose(
        np.diff(contributions[:, 0]),
        np.diff(raw[:, 0]),
    )


def test_centered_pseudo_values_validate_shapes() -> None:
    with pytest.raises(ValueError):
        centered_jackknife_pseudo_values(
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
        )

    with pytest.raises(ValueError):
        centered_jackknife_pseudo_values(
            np.array([1.0]),
            np.array([[1.0]]),
        )


def test_micro_jackknife_result_validates_mean_identity() -> None:
    full = np.array([0.2, -0.1])
    loo = np.array(
        [
            [0.1, -0.2],
            [0.3, 0.0],
        ]
    )
    contributions, raw, adjustment = (
        centered_jackknife_pseudo_values(full, loo)
    )

    result = MicroJackknifeResult(
        market_ids=np.array([1981, 1982]),
        contributions=contributions,
        raw_pseudo_values=raw,
        full_residual=full,
        leave_one_out_residuals=loo,
        centering_adjustment=adjustment,
        full_elapsed_seconds=1.0,
        leave_one_out_elapsed_seconds=np.array([2.0, 3.0]),
    )

    assert result.n_markets == 2
    assert result.n_micro_moments == 2
    assert result.total_elapsed_seconds == pytest.approx(6.0)

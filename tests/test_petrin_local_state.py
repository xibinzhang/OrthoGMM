import numpy as np
import pytest

from orthogmm.applications.petrin.local_state import (
    PetrinLocalState,
    PetrinLocalStateBuilder,
)


def test_local_state_dimensions() -> None:
    state = PetrinLocalState(
        theta=np.array([1.0, 2.0]),
        market_ids=np.array([1981, 1982, 1983]),
        tractable_moments=np.ones((3, 4)),
        demanding_moments=np.ones((3, 2)),
        tractable_jacobian=np.ones((4, 2)),
        demanding_jacobian=np.ones((2, 2)),
        aggregate_elapsed_seconds=1.0,
        micro_elapsed_seconds=2.0,
    )

    assert state.n_markets == 3
    assert state.n_tractable_moments == 4
    assert state.n_demanding_moments == 2
    assert state.parameter_dimension == 2
    assert state.total_elapsed_seconds == pytest.approx(3.0)


def test_local_state_rejects_misaligned_markets() -> None:
    with pytest.raises(ValueError, match="one row per market"):
        PetrinLocalState(
            theta=np.array([1.0]),
            market_ids=np.array([1981, 1982]),
            tractable_moments=np.ones((3, 2)),
            demanding_moments=np.ones((2, 1)),
            tractable_jacobian=np.ones((2, 1)),
            demanding_jacobian=np.ones((1, 1)),
            aggregate_elapsed_seconds=1.0,
            micro_elapsed_seconds=1.0,
        )


def test_active_jacobian_selects_active_block() -> None:
    matrix = np.arange(30, dtype=float).reshape(5, 6)

    selected = PetrinLocalStateBuilder._active_jacobian(
        matrix,
        rows=4,
        parameters=3,
        name="test",
    )

    np.testing.assert_allclose(
        selected,
        matrix[:4, :3],
    )

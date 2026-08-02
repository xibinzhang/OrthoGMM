import numpy as np
import pytest

from orthogmm.moments import (
    MomentData,
    PyBLPMomentBuilder,
)


class FakePyBLPResults:
    def __init__(self):
        self.moments = np.array([[1.0], [2.0], [3.0]])
        self.moments_jacobian = np.arange(
            12,
            dtype=float,
        ).reshape(3, 4)
        self.W = np.eye(3)
        self.moments_covariances = 2.0 * np.eye(3)


def test_moment_data_validates_average() -> None:
    unit_moments = np.array(
        [
            [0.0, 1.0],
            [2.0, 3.0],
        ]
    )

    data = MomentData(
        unit_ids=np.array(["a", "b"]),
        unit_moments=unit_moments,
        average_moments=unit_moments.mean(axis=0),
        jacobian=np.ones((2, 1)),
        weighting=np.eye(2),
    )

    assert data.n_units == 2
    assert data.n_moments == 2
    assert data.n_parameters == 1


def test_moment_data_rejects_inconsistent_average() -> None:
    with pytest.raises(ValueError, match="average_moments"):
        MomentData(
            unit_ids=np.array([1, 2]),
            unit_moments=np.zeros((2, 2)),
            average_moments=np.ones(2),
            jacobian=np.ones((2, 1)),
            weighting=np.eye(2),
        )


def test_pyblp_builder_preserves_public_metadata() -> None:
    results = FakePyBLPResults()
    market_moments = np.array(
        [
            [0.0, 1.0, 2.0],
            [2.0, 3.0, 4.0],
        ]
    )

    data = PyBLPMomentBuilder().build(
        results,
        market_ids=np.array([1981, 1982]),
        market_moments=market_moments,
    )

    np.testing.assert_allclose(
        data.average_moments,
        results.moments.reshape(-1),
    )
    np.testing.assert_allclose(
        data.jacobian,
        results.moments_jacobian,
    )
    np.testing.assert_allclose(data.weighting, results.W)
    np.testing.assert_allclose(
        data.covariance,
        results.moments_covariances,
    )


def test_pyblp_builder_rejects_placeholder_zeros() -> None:
    results = FakePyBLPResults()

    with pytest.raises(ValueError, match="average_moments"):
        PyBLPMomentBuilder().build(
            results,
            market_ids=np.array([1981, 1982]),
            market_moments=np.zeros((2, 3)),
        )


def test_pyblp_metadata_dimensions() -> None:
    metadata = PyBLPMomentBuilder.metadata(
        FakePyBLPResults()
    )

    assert metadata == {
        "n_moments": 3,
        "n_parameters": 4,
    }

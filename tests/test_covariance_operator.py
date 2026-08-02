import numpy as np
import pytest

from orthogmm import CovarianceOperator
from orthogmm.exceptions import ModelContractError


def test_iid_covariance_matches_centered_second_moment() -> None:
    moments = np.array(
        [[1.0, 2.0], [2.0, 0.0], [4.0, 3.0], [5.0, 1.0]]
    )
    result = CovarianceOperator().fit(moments)

    centered = moments - moments.mean(axis=0)
    expected = centered.T @ centered / moments.shape[0]

    np.testing.assert_allclose(result.covariance, expected)
    np.testing.assert_allclose(
        result.weight @ result.covariance,
        np.eye(2),
        rtol=1e-9,
        atol=1e-9,
    )
    assert result.covariance_type == "iid"
    assert result.n_units == 4
    assert result.n_moments == 2
    assert result.n_clusters is None


def test_cluster_covariance_matches_centered_cluster_sums() -> None:
    moments = np.array(
        [
            [1.0, 0.0], [2.0, 1.0],
            [0.0, 2.0], [1.0, 3.0],
            [4.0, 1.0], [2.0, 2.0],
        ]
    )
    clusters = np.array([0, 0, 1, 1, 2, 2])

    result = CovarianceOperator().fit(
        moments,
        covariance_type="cluster",
        clusters=clusters,
    )

    cluster_sums = np.array(
        [
            moments[clusters == cluster].sum(axis=0)
            for cluster in np.unique(clusters)
        ]
    )
    centered = cluster_sums - cluster_sums.mean(axis=0)
    expected = centered.T @ centered / moments.shape[0]

    np.testing.assert_allclose(result.covariance, expected)
    assert result.n_clusters == 3


def test_one_dimensional_moments_are_supported() -> None:
    result = CovarianceOperator().fit(
        np.array([1.0, 2.0, 4.0, 8.0])
    )
    assert result.covariance.shape == (1, 1)
    assert result.weight.shape == (1, 1)


def test_operator_properties_require_fit() -> None:
    operator = CovarianceOperator()

    with pytest.raises(RuntimeError):
        _ = operator.covariance_

    with pytest.raises(RuntimeError):
        _ = operator.weight_


def test_operator_properties_match_result() -> None:
    moments = np.array(
        [[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]]
    )
    operator = CovarianceOperator(ridge=1e-8)
    result = operator.fit(moments)

    np.testing.assert_allclose(
        operator.covariance_,
        result.covariance,
    )
    np.testing.assert_allclose(
        operator.weight_,
        result.weight,
    )
    assert operator.condition_number_ == result.condition_number
    assert operator.effective_rank_ == result.effective_rank
    assert operator.ridge_ == result.ridge


def test_cluster_covariance_requires_valid_cluster_ids() -> None:
    moments = np.ones((4, 2))

    with pytest.raises(ModelContractError):
        CovarianceOperator().fit(
            moments,
            covariance_type="cluster",
        )

    with pytest.raises(ModelContractError):
        CovarianceOperator().fit(
            moments,
            covariance_type="cluster",
            clusters=np.array([0, 0, 1]),
        )

    with pytest.raises(ModelContractError):
        CovarianceOperator().fit(
            moments,
            covariance_type="cluster",
            clusters=np.zeros(4),
        )


def test_invalid_moment_array_is_rejected() -> None:
    with pytest.raises(ModelContractError):
        CovarianceOperator().fit(
            np.array([[np.nan], [1.0]])
        )

    with pytest.raises(ModelContractError):
        CovarianceOperator().fit(
            np.array([[1.0]])
        )


def test_regularization_diagnostics_are_recorded() -> None:
    moments = np.column_stack(
        [np.arange(1.0, 6.0), np.arange(1.0, 6.0)]
    )
    result = CovarianceOperator(ridge=1e-8).fit(moments)

    assert result.ridge > 0.0
    assert np.isfinite(result.condition_number)
    assert result.effective_rank == 2
    assert np.all(np.isfinite(result.weight))

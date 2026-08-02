import numpy as np
import pytest

from orthogmm.operators.basis import TractableMomentBasis


def test_basis_detects_centered_rank() -> None:
    rng = np.random.default_rng(1)
    scores = rng.normal(size=(20, 3))
    loadings = rng.normal(size=(3, 8))
    g = scores @ loadings
    G = rng.normal(size=(8, 2))

    result = TractableMomentBasis().fit(g, G)

    assert result.empirical_rank == 3
    assert result.retained_rank == 3
    assert result.reduced_moments.shape == (20, 3)
    assert result.reduced_jacobian.shape == (3, 2)


def test_basis_respects_fixed_rank() -> None:
    rng = np.random.default_rng(2)
    g = rng.normal(size=(30, 6))
    G = rng.normal(size=(6, 2))

    result = TractableMomentBasis(rank=2).fit(g, G)

    assert result.retained_rank == 2
    assert result.explained_variance_ratio < 1.0


def test_basis_selects_explained_variance() -> None:
    rng = np.random.default_rng(3)
    first = rng.normal(size=(50, 1))
    noise = rng.normal(scale=0.01, size=(50, 4))
    g = np.column_stack([first, noise])
    G = rng.normal(size=(5, 2))

    result = TractableMomentBasis(
        explained_variance=0.95,
    ).fit(g, G)

    assert result.retained_rank == 1
    assert result.explained_variance_ratio >= 0.95


def test_basis_is_orthonormal() -> None:
    rng = np.random.default_rng(4)
    g = rng.normal(size=(25, 7))
    G = rng.normal(size=(7, 3))

    result = TractableMomentBasis(rank=4).fit(g, G)

    np.testing.assert_allclose(
        result.basis.T @ result.basis,
        np.eye(4),
        atol=1e-12,
    )


def test_basis_rejects_conflicting_selection_rules() -> None:
    with pytest.raises(ValueError):
        TractableMomentBasis(
            rank=2,
            explained_variance=0.9,
        )

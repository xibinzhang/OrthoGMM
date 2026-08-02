import numpy as np

from orthogmm.operators.projected_information import (
    ProjectedInformationOperator,
)


def make_inputs(seed: int = 123):
    rng = np.random.default_rng(seed)
    n = 500
    qg = 4
    qh = 3
    p = 2

    g = rng.normal(size=(n, qg))
    B_true = np.array(
        [
            [0.4, -0.2, 0.1, 0.0],
            [0.0, 0.3, -0.1, 0.2],
            [0.1, 0.0, 0.2, -0.3],
        ]
    )
    noise = rng.normal(scale=0.5, size=(n, qh))
    h = g @ B_true.T + noise

    G = rng.normal(size=(qg, p))
    H = rng.normal(size=(qh, p))
    return g, h, G, H


def test_reduced_projection_dimensions() -> None:
    g, h, G, H = make_inputs()

    result = ProjectedInformationOperator(rank=3).fit(
        g,
        h,
        G,
        H,
    )

    assert result.projection.shape == (3, 4)
    assert result.reduced_projection.shape == (3, 3)
    assert result.omega_gg.shape == (3, 3)
    assert result.residual_jacobian.shape == (3, 2)
    assert result.information.shape == (2, 2)


def test_full_rank_projection_is_orthogonal() -> None:
    g, h, G, H = make_inputs()

    result = ProjectedInformationOperator().fit(
        g,
        h,
        G,
        H,
    )

    reduced = result.basis_result.reduced_moments
    reduced_centered = reduced - reduced.mean(axis=0)
    residual_centered = (
        result.residual_moments
        - result.residual_moments.mean(axis=0)
    )
    cross = residual_centered.T @ reduced_centered / g.shape[0]

    assert np.linalg.norm(cross) < 1e-10

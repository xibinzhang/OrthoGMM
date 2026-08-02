import numpy as np

from orthogmm.estimators.sequential import (
    SequentialProjectionCorrection,
)
from orthogmm.operators import ProjectedInformationOperator


def test_linear_system_moves_to_zero() -> None:
    rng = np.random.default_rng(1234)
    n = 1000
    theta = np.array([0.4, -0.2])

    G = np.eye(2)
    H = np.zeros((1, 2))

    noise_g = rng.normal(scale=0.05, size=(n, 2))
    noise_h = rng.normal(scale=0.05, size=(n, 1))

    # Local moments satisfy g(theta) approximately theta.
    g = theta + noise_g
    h = noise_h

    projected = ProjectedInformationOperator(
        rank=2,
        ridge=1e-10,
        center=True,
    ).fit(g, h, G, H)

    result = SequentialProjectionCorrection().apply(
        theta,
        projected,
    )

    assert np.linalg.norm(result.theta_updated) < 0.02


def test_radius_limits_step_norm() -> None:
    rng = np.random.default_rng(7)
    theta = np.array([1.0, -1.0])
    g = theta + rng.normal(scale=0.1, size=(500, 2))
    h = rng.normal(size=(500, 1))
    G = np.eye(2)
    H = np.zeros((1, 2))

    projected = ProjectedInformationOperator(
        rank=2,
        ridge=1e-8,
    ).fit(g, h, G, H)

    result = SequentialProjectionCorrection(
        damping=1.0,
        radius=0.05,
    ).apply(theta, projected)

    assert result.radius_clipped
    assert result.applied_step_norm <= 0.05 + 1e-12


def test_damping_scales_unclipped_direction() -> None:
    rng = np.random.default_rng(8)
    theta = np.array([0.2, 0.1])
    g = theta + rng.normal(scale=0.1, size=(500, 2))
    h = rng.normal(size=(500, 1))
    G = np.eye(2)
    H = np.zeros((1, 2))

    projected = ProjectedInformationOperator(
        rank=2,
        ridge=1e-8,
    ).fit(g, h, G, H)

    result = SequentialProjectionCorrection(
        damping=0.25,
    ).apply(theta, projected)

    np.testing.assert_allclose(
        result.applied_step,
        0.25 * result.raw_direction,
    )

import numpy as np

from orthogmm.estimators.seip import (
    SequentialEfficientInfluenceProjection,
)


def test_seip_returns_finite_update() -> None:
    rng = np.random.default_rng(42)
    n = 400
    qg = 4
    qh = 2
    p = 2

    theta = np.array([0.2, -0.1])
    G = rng.normal(size=(qg, p))
    H = rng.normal(size=(qh, p))

    g = rng.normal(size=(n, qg)) + 0.05
    h = 0.3 * g[:, :2] + rng.normal(scale=0.2, size=(n, qh))

    result = SequentialEfficientInfluenceProjection(
        rank=3,
        radius=0.5,
        metric_type="information",
    ).fit(theta, g, h, G, H)

    assert result.theta_updated.shape == (p,)
    assert np.all(np.isfinite(result.theta_updated))
    assert result.retained_rank == 3
    assert result.trust_region.metric_type == "information"


def test_seip_information_radius_is_respected() -> None:
    rng = np.random.default_rng(7)
    n = 500
    p = 2

    theta = np.array([1.0, 2.0])
    g = rng.normal(size=(n, 3)) + np.array([0.5, -0.4, 0.3])
    h = rng.normal(size=(n, 2)) + np.array([0.2, -0.1])
    G = rng.normal(size=(3, p))
    H = rng.normal(size=(2, p))

    radius = 0.25
    result = SequentialEfficientInfluenceProjection(
        rank=3,
        radius=radius,
        metric_type="information",
    ).fit(theta, g, h, G, H)

    step = result.step
    J = result.projected_information.information
    metric_norm = float(np.sqrt(step @ J @ step))

    assert metric_norm <= radius * (1.0 + 1e-8)


def test_none_radius_uses_sqrt_parameter_dimension() -> None:
    rng = np.random.default_rng(9)
    n = 300
    p = 3

    theta = np.zeros(p)
    g = rng.normal(size=(n, 4)) + 0.1
    h = rng.normal(size=(n, 2)) + 0.1
    G = rng.normal(size=(4, p))
    H = rng.normal(size=(2, p))

    result = SequentialEfficientInfluenceProjection(
        rank=3,
        radius=None,
    ).fit(theta, g, h, G, H)

    assert result.trust_region.radius == np.sqrt(p)

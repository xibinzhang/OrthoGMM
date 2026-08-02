import numpy as np

from orthogmm import (
    FidelityConfig,
    MultiFidelityBLPModel,
    fit_full_gmm,
    fit_seip,
    fit_tractable_gmm,
)


class ToyBLPModel(MultiFidelityBLPModel):
    """Small deterministic model for testing the BLP fidelity partition."""

    def __init__(self, seed: int = 123, n_markets: int = 200):
        rng = np.random.default_rng(seed)

        self.low_fidelity = FidelityConfig(
            name="low",
            draws=10,
            contraction_tolerance=1e-6,
            max_iterations=100,
            seed=1,
        )
        self.high_fidelity = FidelityConfig(
            name="high",
            draws=100,
            contraction_tolerance=1e-12,
            max_iterations=1000,
            seed=1,
        )

        self.x = rng.normal(size=(n_markets, 2))
        self.zg = rng.normal(size=(n_markets, 2))
        self.zh = rng.normal(size=(n_markets, 2))
        self.truth = np.array([1.0, -0.5])
        self.y = self.x @ self.truth + rng.normal(
            scale=0.25,
            size=n_markets,
        )

    def moments_at_fidelity(self, theta, fidelity):
        residual = self.y - self.x @ np.asarray(theta)

        low = self.zg * residual[:, None]

        if fidelity.name == "low":
            return low

        if fidelity.name == "high":
            return low + self.zh * residual[:, None]

        raise ValueError(f"Unknown fidelity: {fidelity.name}")


def test_blp_partition_is_high_minus_low() -> None:
    model = ToyBLPModel()
    theta = np.array([0.9, -0.4])

    low = model.moments_at_fidelity(
        theta,
        model.low_fidelity,
    )
    high = model.moments_at_fidelity(
        theta,
        model.high_fidelity,
    )

    np.testing.assert_allclose(
        model.tractable_moments(theta),
        low,
    )
    np.testing.assert_allclose(
        model.demanding_moments(theta),
        high - low,
    )


def test_blp_adapter_supports_all_estimators() -> None:
    model = ToyBLPModel(n_markets=500)
    theta0 = np.zeros(2)

    initial = fit_tractable_gmm(model, theta0)
    full = fit_full_gmm(model, theta0)
    sop = fit_seip(model, theta0)

    assert initial.theta.shape == (2,)
    assert full.theta.shape == (2,)
    assert sop.theta.shape == (2,)
    assert np.all(np.isfinite(initial.theta))
    assert np.all(np.isfinite(full.theta))
    assert np.all(np.isfinite(sop.theta))


def test_full_and_sop_are_close_for_toy_blp() -> None:
    model = ToyBLPModel(seed=456, n_markets=1000)
    theta0 = np.zeros(2)

    full = fit_full_gmm(model, theta0)
    sop = fit_seip(model, theta0)

    assert np.linalg.norm(full.theta - sop.theta) < 0.05
    assert sop.counts.demanding_moments_total < (
        full.counts.demanding_moments_total
    )

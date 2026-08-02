import numpy as np

from orthogmm import fit_full_gmm


class OveridentifiedLinearModel:
    def __init__(self, seed: int = 123, n: int = 1000):
        rng = np.random.default_rng(seed)
        self.z = rng.normal(size=(n, 4))
        self.x = np.column_stack(
            [
                self.z[:, 0] + 0.5 * self.z[:, 2]
                + rng.normal(scale=0.4, size=n),
                self.z[:, 1] + 0.5 * self.z[:, 3]
                + rng.normal(scale=0.4, size=n),
            ]
        )
        self.truth = np.array([1.0, -0.5])
        self.y = self.x @ self.truth + rng.normal(size=n)
        self.zg = self.z[:, :2]
        self.zh = self.z[:, 2:]

    def residual(self, theta):
        return self.y - self.x @ theta

    def tractable_moments(self, theta):
        return self.zg * self.residual(theta)[:, None]

    def demanding_moments(self, theta):
        return self.zh * self.residual(theta)[:, None]

    def tractable_jacobian(self, theta):
        return -(self.zg.T @ self.x) / len(self.y)

    def demanding_jacobian(self, theta):
        return -(self.zh.T @ self.x) / len(self.y)


def test_full_gmm_is_two_step() -> None:
    model = OveridentifiedLinearModel()
    result = fit_full_gmm(
        model,
        theta0=np.zeros(2),
    )

    assert result.success
    assert result.preliminary_theta.shape == (2,)
    assert result.theta.shape == (2,)
    assert isinstance(result.optimizer_result, dict)
    assert "stage_one" in result.optimizer_result
    assert "stage_two" in result.optimizer_result
    assert "efficient_weight" in result.optimizer_result


def test_two_step_full_gmm_changes_weight() -> None:
    model = OveridentifiedLinearModel(seed=321)
    result = fit_full_gmm(
        model,
        theta0=np.zeros(2),
    )

    identity = result.optimizer_result["stage_one_weight"]
    efficient = result.optimizer_result["efficient_weight"]

    assert identity.shape == efficient.shape
    assert not np.allclose(identity, efficient)


def test_two_step_full_gmm_records_both_optimizations() -> None:
    model = OveridentifiedLinearModel(seed=456)
    result = fit_full_gmm(
        model,
        theta0=np.zeros(2),
    )

    stage_one = result.optimizer_result["stage_one"]
    stage_two = result.optimizer_result["stage_two"]

    assert stage_one.success
    assert stage_two.success
    assert result.counts.tractable_objective >= (
        stage_one.nfev + stage_two.nfev
    )


def test_two_step_full_gmm_is_close_to_truth() -> None:
    model = OveridentifiedLinearModel(seed=789, n=2500)
    result = fit_full_gmm(
        model,
        theta0=np.zeros(2),
    )

    assert np.linalg.norm(result.theta - model.truth) < 0.12
    assert np.all(np.isfinite(result.standard_errors))
    assert result.covariance.shape == (2, 2)


def test_second_stage_uses_preliminary_estimate() -> None:
    model = OveridentifiedLinearModel(seed=654)
    result = fit_full_gmm(
        model,
        theta0=np.zeros(2),
    )

    stage_two = result.optimizer_result["stage_two"]

    np.testing.assert_allclose(
        stage_two.x,
        result.theta,
    )
    assert np.linalg.norm(
        result.preliminary_theta - result.theta
    ) > 1e-10

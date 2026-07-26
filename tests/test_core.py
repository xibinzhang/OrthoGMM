import numpy as np

from orthogmm import fit_full_gmm, fit_seip, fit_tractable_gmm


class LinearModel:
    def __init__(self, y, x, zg, zh):
        self.y, self.x, self.zg, self.zh = y, x, zg, zh

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


def make_model(seed=8, n=2500):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 4))
    x = np.column_stack((z[:, 0] + 0.4*z[:, 2] + rng.normal(scale=.4, size=n),
                         z[:, 1] + 0.4*z[:, 3] + rng.normal(scale=.4, size=n)))
    beta = np.array([1.0, -0.5])
    y = x @ beta + rng.normal(size=n)
    return LinearModel(y, x, z[:, :2], z[:, 2:]), beta


def test_estimators_are_close_to_truth():
    model, truth = make_model()
    start = np.zeros(2)
    tr = fit_tractable_gmm(model, start)
    full = fit_full_gmm(model, start)
    seip = fit_seip(model, start)
    assert np.linalg.norm(tr.theta - truth) < 0.12
    assert np.linalg.norm(full.theta - truth) < 0.12
    assert np.linalg.norm(seip.theta - truth) < 0.12
    assert seip.covariance.shape == (2, 2)
    assert seip.projection.shape == (2, 2)


def test_projection_is_orthogonal_in_sample():
    model, _ = make_model(seed=11)
    result = fit_seip(model, np.zeros(2))
    assert np.linalg.norm(result.orthogonality_residual) < 1e-8


def test_user_supplied_preliminary_estimator():
    model, truth = make_model(seed=13)
    result = fit_seip(model, np.zeros(2), preliminary_theta=truth)
    assert result.counts.tractable_objective == 0
    assert result.theta.shape == truth.shape

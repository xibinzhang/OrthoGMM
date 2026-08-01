import numpy as np

from orthogmm import FullGMM, OrthogonalProjection, SOPEstimator, TractableGMM


class TinyModel:
    def __init__(self):
        self.x = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        self.y = 2.0 * self.x[:, 0] + np.array([-.2, .1, .2, -.1, 0.0])
        self.zg = np.column_stack([np.ones(5), self.x[:, 0]])
        self.zh = np.column_stack([self.x[:, 0] ** 2, self.x[:, 0] ** 3])

    def _u(self, theta):
        return self.y - self.x[:, 0] * theta[0]

    def tractable_moments(self, theta):
        return self.zg * self._u(theta)[:, None]

    def demanding_moments(self, theta):
        return self.zh * self._u(theta)[:, None]

    def tractable_jacobian(self, theta):
        return -(self.zg.T @ self.x) / len(self.y)

    def demanding_jacobian(self, theta):
        return -(self.zh.T @ self.x) / len(self.y)


def test_estimator_classes_preserve_functional_api():
    model = TinyModel()
    theta0 = np.array([0.0])
    assert TractableGMM().fit(model, theta0).theta.shape == (1,)
    assert FullGMM().fit(model, theta0).theta.shape == (1,)
    assert SOPEstimator().fit(model, theta0).theta.shape == (1,)


def test_projection_operator_constructs_section_3_objects():
    model = TinyModel()
    theta = np.array([2.0])
    g = model.tractable_moments(theta)
    h = model.demanding_moments(theta)
    G = model.tractable_jacobian(theta)
    H = model.demanding_jacobian(theta)
    out = OrthogonalProjection(ridge=1e-10).fit(g, h, G, H)
    assert out.coefficient.shape == (2, 2)
    assert out.residualized_jacobian.shape == (2, 1)
    assert out.information.shape == (1, 1)
    assert np.linalg.norm(out.orthogonality_residual) < 1e-6

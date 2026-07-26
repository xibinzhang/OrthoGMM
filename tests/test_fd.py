import numpy as np

from orthogmm.differentiation import finite_difference_jacobian


def test_finite_difference_jacobian():
    def f(theta):
        return np.array([theta[0] ** 2 + theta[1], np.sin(theta[1])])

    theta = np.array([2.0, 0.3])
    got = finite_difference_jacobian(f, theta)
    expected = np.array([[4.0, 1.0], [0.0, np.cos(0.3)]])
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

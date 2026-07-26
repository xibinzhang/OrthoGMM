"""Small linear-IV example comparing tractable, full, and SEIP estimators."""

from __future__ import annotations

import numpy as np

from orthogmm import fit_full_gmm, fit_seip, fit_tractable_gmm


class LinearIVModel:
    def __init__(self, y: np.ndarray, x: np.ndarray, z_g: np.ndarray, z_h: np.ndarray):
        self.y = y
        self.x = x
        self.z_g = z_g
        self.z_h = z_h

    def _residual(self, theta: np.ndarray) -> np.ndarray:
        return self.y - self.x @ theta

    def tractable_moments(self, theta: np.ndarray) -> np.ndarray:
        return self.z_g * self._residual(theta)[:, None]

    def demanding_moments(self, theta: np.ndarray) -> np.ndarray:
        # In a structural application this call would be expensive.
        return self.z_h * self._residual(theta)[:, None]

    def tractable_jacobian(self, theta: np.ndarray) -> np.ndarray:
        del theta
        return -(self.z_g.T @ self.x) / self.x.shape[0]

    def demanding_jacobian(self, theta: np.ndarray) -> np.ndarray:
        del theta
        return -(self.z_h.T @ self.x) / self.x.shape[0]


rng = np.random.default_rng(20260726)
n = 3000
z = rng.normal(size=(n, 4))
x = np.column_stack([
    1.0 + 0.7 * z[:, 0] + 0.3 * z[:, 2] + rng.normal(scale=0.5, size=n),
    -0.4 + 0.6 * z[:, 1] + 0.3 * z[:, 3] + rng.normal(scale=0.5, size=n),
])
theta_true = np.array([1.0, -0.5])
y = x @ theta_true + rng.normal(scale=1.0, size=n)
model = LinearIVModel(y, x, z[:, :2], z[:, 2:])
start = np.zeros(2)

for result in (
    fit_tractable_gmm(model, start),
    fit_full_gmm(model, start),
    fit_seip(model, start),
):
    print("=" * 72)
    print(result.summary())

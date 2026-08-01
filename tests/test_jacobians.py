import numpy as np

from orthogmm.jacobians import (
    AnalyticalJacobian,
    FallbackJacobian,
    FiniteDifferenceJacobian,
    JacobianContext,
)
from orthogmm.types import EvaluationCounts


class AnalyticalModel:
    def __init__(self):
        self.x = np.array([1.0, 2.0, 3.0, 4.0])
        self.zg = np.column_stack(
            [np.ones(4), self.x]
        )
        self.zh = np.column_stack(
            [self.x**2, self.x**3]
        )

    def tractable_moments(self, theta):
        residual = 2.0 * self.x - theta[0] * self.x
        return self.zg * residual[:, None]

    def demanding_moments(self, theta):
        residual = 2.0 * self.x - theta[0] * self.x
        return self.zh * residual[:, None]

    def tractable_jacobian(self, theta):
        del theta
        return -(self.zg.T @ self.x[:, None]) / self.x.size

    def demanding_jacobian(self, theta):
        del theta
        return -(self.zh.T @ self.x[:, None]) / self.x.size


class NumericalModel(AnalyticalModel):
    def tractable_jacobian(self, theta):
        raise NotImplementedError

    def demanding_jacobian(self, theta):
        raise NotImplementedError


def make_context(model, kind):
    return JacobianContext(
        model=model,
        theta=np.array([1.5]),
        moment_kind=kind,
        counts=EvaluationCounts(),
        bounds=None,
    )


def test_analytical_jacobian_returns_model_derivative():
    model = AnalyticalModel()
    context = make_context(model, "tractable")

    result = AnalyticalJacobian().compute(context)
    expected = model.tractable_jacobian(context.theta)

    np.testing.assert_allclose(result, expected)
    assert context.counts.tractable_jacobian == 1


def test_finite_difference_matches_analytical_tractable_jacobian():
    model = AnalyticalModel()
    context = make_context(model, "tractable")

    result = FiniteDifferenceJacobian().compute(context)
    expected = model.tractable_jacobian(context.theta)

    np.testing.assert_allclose(
        result,
        expected,
        rtol=1e-6,
        atol=1e-8,
    )
    assert context.counts.tractable_moments > 0


def test_finite_difference_matches_analytical_demanding_jacobian():
    model = AnalyticalModel()
    context = make_context(model, "demanding")

    result = FiniteDifferenceJacobian().compute(context)
    expected = model.demanding_jacobian(context.theta)

    np.testing.assert_allclose(
        result,
        expected,
        rtol=1e-6,
        atol=1e-8,
    )
    assert context.counts.demanding_moments_derivative > 0


def test_fallback_prefers_analytical_jacobian():
    model = AnalyticalModel()
    context = make_context(model, "tractable")

    result = FallbackJacobian().compute(context)
    expected = model.tractable_jacobian(context.theta)

    np.testing.assert_allclose(result, expected)
    assert context.counts.tractable_jacobian == 1
    assert context.counts.tractable_moments == 0


def test_fallback_uses_finite_difference_when_needed():
    model = NumericalModel()
    context = make_context(model, "demanding")

    result = FallbackJacobian().compute(context)
    expected = AnalyticalModel().demanding_jacobian(
        context.theta
    )

    np.testing.assert_allclose(
        result,
        expected,
        rtol=1e-6,
        atol=1e-8,
    )
    assert context.counts.demanding_jacobian == 0
    assert context.counts.demanding_moments_derivative > 0

import numpy as np

from orthogmm import (
    FullGMM,
    OrthogonalProjection,
    SOPEstimator,
    TractableGMM,
    fit_seip,
)


class TinyModel:
    def __init__(self):
        self.x = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        self.y = (
            2.0 * self.x[:, 0]
            + np.array([-0.2, 0.1, 0.2, -0.1, 0.0])
        )
        self.zg = np.column_stack(
            [np.ones(5), self.x[:, 0]]
        )
        self.zh = np.column_stack(
            [self.x[:, 0] ** 2, self.x[:, 0] ** 3]
        )

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

    out = OrthogonalProjection(ridge=1e-10).fit(
        g,
        h,
        G,
        H,
    )

    assert out.coefficient.shape == (2, 2)
    assert out.residualized_jacobian.shape == (2, 1)
    assert out.information.shape == (1, 1)
    assert np.linalg.norm(out.orthogonality_residual) < 1e-6


def test_fit_seip_uses_canonical_projection_operator():
    model = TinyModel()
    preliminary = np.array([2.0])
    ridge = 1e-10

    g = model.tractable_moments(preliminary)
    h = model.demanding_moments(preliminary)
    G = model.tractable_jacobian(preliminary)
    H = model.demanding_jacobian(preliminary)
    tractable_weight = np.eye(g.shape[1])

    projection = OrthogonalProjection(ridge=ridge).fit(
        g,
        h,
        G,
        H,
        covariance_type="iid",
        tractable_weight=tractable_weight,
    )

    result = fit_seip(
        model,
        theta0=np.array([0.0]),
        preliminary_theta=preliminary,
        covariance_type="iid",
        tractable_weight=tractable_weight,
        ridge=ridge,
    )

    np.testing.assert_allclose(
        result.projection,
        projection.coefficient,
    )
    np.testing.assert_allclose(
        result.residual_covariance,
        projection.residual_covariance,
    )
    np.testing.assert_allclose(
        result.R,
        projection.residualized_jacobian,
    )
    np.testing.assert_allclose(
        result.information,
        projection.information,
    )
    np.testing.assert_allclose(
        result.orthogonality_residual,
        projection.orthogonality_residual,
    )

    expected_residual_update = -np.linalg.solve(
        projection.information,
        projection.residual_score,
    )
    expected_full_update = -np.linalg.solve(
        projection.information,
        projection.projected_score,
    )

    np.testing.assert_allclose(
        result.residual_only_update,
        expected_residual_update,
    )
    np.testing.assert_allclose(
        result.full_score_update,
        expected_full_update,
    )
    np.testing.assert_allclose(
        result.update,
        expected_residual_update,
    )
    np.testing.assert_allclose(
        result.theta,
        preliminary + expected_residual_update,
    )
    np.testing.assert_allclose(
        result.full_score_update
        - result.residual_only_update,
        -np.linalg.solve(
            projection.information,
            projection.tractable_score,
        ),
    )


def test_projection_operator_supports_cluster_covariance():
    model = TinyModel()
    theta = np.array([2.0])
    clusters = np.array([0, 0, 1, 1, 2])

    g = model.tractable_moments(theta)
    h = model.demanding_moments(theta)
    G = model.tractable_jacobian(theta)
    H = model.demanding_jacobian(theta)

    out = OrthogonalProjection(ridge=1e-10).fit(
        g,
        h,
        G,
        H,
        covariance_type="cluster",
        clusters=clusters,
    )

    assert out.coefficient.shape == (2, 2)
    assert out.residual_covariance.shape == (2, 2)
    assert out.residualized_jacobian.shape == (2, 1)
    assert out.information.shape == (1, 1)
    assert np.all(np.isfinite(out.information))


def test_matched_tractable_foc_gives_residual_only_cancellation():
    model = TinyModel()
    theta_zero = np.array([0.0])
    tractable_weight = np.eye(2)

    g_zero = model.tractable_moments(theta_zero)
    gbar_zero = g_zero.mean(axis=0)
    G = model.tractable_jacobian(theta_zero)

    # Exact minimizer of the quadratic tractable criterion:
    # gbar(theta) = gbar(0) + G theta.
    theta_tractable = -np.linalg.solve(
        G.T @ tractable_weight @ G,
        G.T @ tractable_weight @ gbar_zero,
    )

    result = fit_seip(
        model,
        theta0=theta_zero,
        preliminary_theta=theta_tractable,
        tractable_weight=tractable_weight,
        covariance_type="iid",
        ridge=1e-10,
        damping=1.0,
    )

    assert result.tractable_foc_norm is not None
    assert result.update_difference_norm is not None
    assert result.tractable_foc_norm < 1e-11
    assert result.update_difference_norm < 1e-10

    np.testing.assert_allclose(
        result.full_score_update,
        result.residual_only_update,
        rtol=1e-9,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        result.update,
        result.residual_only_update,
        rtol=1e-12,
        atol=1e-12,
    )


def test_mismatched_weight_prevents_residual_only_cancellation():
    model = TinyModel()
    theta_zero = np.array([0.0])

    weight_localization = np.eye(2)
    weight_update = np.array([
        [2.0, 0.0],
        [0.0, 0.5],
    ])

    g_zero = model.tractable_moments(theta_zero)
    gbar_zero = g_zero.mean(axis=0)
    G = model.tractable_jacobian(theta_zero)

    theta_tractable = -np.linalg.solve(
        G.T @ weight_localization @ G,
        G.T @ weight_localization @ gbar_zero,
    )

    g = model.tractable_moments(theta_tractable)
    h = model.demanding_moments(theta_tractable)
    H = model.demanding_jacobian(theta_tractable)

    projection = OrthogonalProjection(ridge=1e-10).fit(
        g,
        h,
        G,
        H,
        covariance_type="iid",
        tractable_weight=weight_update,
    )

    full_update = -np.linalg.solve(
        projection.information,
        projection.projected_score,
    )
    residual_update = -np.linalg.solve(
        projection.information,
        projection.residual_score,
    )

    assert np.linalg.norm(projection.tractable_score) > 1e-8
    assert np.linalg.norm(full_update - residual_update) > 1e-8

    np.testing.assert_allclose(
        full_update - residual_update,
        -np.linalg.solve(
            projection.information,
            projection.tractable_score,
        ),
        rtol=1e-10,
        atol=1e-11,
    )


def test_default_sop_uses_two_step_matched_weight_localization():
    model = TinyModel()
    theta0 = np.array([0.0])

    result = fit_seip(
        model,
        theta0,
        covariance_type="iid",
        ridge=1e-10,
        damping=1.0,
    )

    assert result.initial_tractable_theta is not None
    assert result.tractable_weight is not None
    assert isinstance(result.optimizer_result, dict)

    stage_one = result.optimizer_result["stage_one"]
    stage_two = result.optimizer_result["stage_two"]

    assert stage_one is not None
    assert stage_two is not None

    np.testing.assert_allclose(
        result.initial_tractable_theta,
        stage_one.x,
    )
    np.testing.assert_allclose(
        result.preliminary_theta,
        stage_two.x,
    )
    np.testing.assert_allclose(
        result.tractable_weight,
        result.optimizer_result["tractable_weight"],
    )

    g = model.tractable_moments(result.preliminary_theta)
    G = model.tractable_jacobian(result.preliminary_theta)
    matched_score = (
        G.T
        @ result.tractable_weight
        @ g.mean(axis=0)
    )

    np.testing.assert_allclose(
        result.tractable_score,
        matched_score,
    )
    assert np.linalg.norm(matched_score) < 1e-5

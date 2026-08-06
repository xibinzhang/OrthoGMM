from types import SimpleNamespace

import numpy as np

from orthogmm.applications.petrin.local_state import PetrinLocalState
from orthogmm.applications.petrin.sop import PetrinResidualOnlySOP


class FakeModel:
    def __init__(self, g: np.ndarray, weight: np.ndarray):
        self.g = g
        self.weight = weight

    def evaluate_aggregate(self, theta):
        results = SimpleNamespace(
            W=self.weight,
            moments=self.g.mean(axis=0),
        )
        pyblp = SimpleNamespace(
            results=results,
            elapsed_seconds=0.0,
        )
        return SimpleNamespace(pyblp=pyblp)


class FakeStateBuilder:
    def __init__(self, state: PetrinLocalState):
        self.state = state
        self.calls = 0

    def build(self, theta):
        self.calls += 1
        return self.state


def build_fixture():
    theta = np.array([0.25])

    g = np.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
        ]
    )
    h = np.array([[1.0], [0.5], [1.5], [0.8], [1.2], [0.7]])

    # The matched tractable score is exactly zero because gbar = 0.
    G = np.array([[1.0], [1.0]])
    H = np.array([[0.5]])

    state = PetrinLocalState(
        theta=theta,
        market_ids=np.arange(g.shape[0]),
        tractable_moments=g,
        demanding_moments=h,
        tractable_jacobian=G,
        demanding_jacobian=H,
        aggregate_elapsed_seconds=0.0,
        micro_elapsed_seconds=0.0,
    )
    weight = np.array([[2.0, 0.2], [0.2, 1.5]])
    return theta, g, state, weight


def test_matched_weight_full_and_residual_updates_coincide():
    theta, g, state, weight = build_fixture()
    builder = FakeStateBuilder(state)

    result = PetrinResidualOnlySOP(
        FakeModel(g, weight),
        ridge=1e-8,
        radius=1.0,
        state_builder=builder,
    ).fit(theta)

    np.testing.assert_allclose(
        result.tractable_weight,
        weight,
    )
    np.testing.assert_allclose(
        result.tractable_score,
        np.zeros_like(result.tractable_score),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.full_update,
        result.residual_only_update,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.update_difference,
        np.zeros_like(result.update_difference),
        atol=1e-12,
    )
    assert builder.calls == 1


def test_trust_region_uses_residual_score():
    theta, g, state, weight = build_fixture()

    result = PetrinResidualOnlySOP(
        FakeModel(g, weight),
        ridge=1e-8,
        radius=1.0,
        state_builder=FakeStateBuilder(state),
    ).fit(theta)

    # With exact matched-weight cancellation, the applied residual-score step
    # is also the step that would be obtained from the full transformed score.
    assert np.isfinite(result.applied_step_norm)
    assert result.applied_step.shape == theta.shape
    np.testing.assert_allclose(
        result.projected_score,
        result.residual_score,
        atol=1e-12,
    )

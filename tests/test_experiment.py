from types import SimpleNamespace

import numpy as np

from orthogmm.simulation import Experiment, LinearIVDesign
from orthogmm.types import EvaluationCounts


def test_experiment_extracts_nested_orthogmm_counts() -> None:
    design = LinearIVDesign(n=50, seed=123)

    def runner(data):
        counts = EvaluationCounts(
            tractable_objective=7,
            demanding_moments_projection=2,
            demanding_moments_derivative=3,
        )
        return SimpleNamespace(
            theta=data["theta_true"].copy(),
            standard_errors=np.array([0.1, 0.2]),
            success=True,
            message="ok",
            counts=counts,
        )

    results = Experiment(
        design=design,
        estimators={"test": runner},
        repetitions=3,
        seed=99,
    ).run()

    assert len(results.records) == 3
    assert all(
        record.objective_evaluations == 7
        for record in results.records
    )
    assert all(
        record.demanding_evaluations == 5
        for record in results.records
    )


def test_experiment_summary_computes_statistical_metrics() -> None:
    design = LinearIVDesign(n=50, seed=123)

    offsets = {
        0: np.array([0.1, -0.2]),
        1: np.array([-0.1, 0.2]),
    }
    call_count = {"value": 0}

    def runner(data):
        index = call_count["value"]
        call_count["value"] += 1
        offset = offsets[index]
        return {
            "theta": data["theta_true"] + offset,
            "standard_errors": np.array([0.1, 0.2]),
            "success": True,
            "objective_evaluations": 4,
            "demanding_evaluations": 2,
        }

    results = Experiment(
        design=design,
        estimators={"test": runner},
        repetitions=2,
        seed=99,
    ).run()

    summaries = results.summarize()
    first = summaries[0]
    second = summaries[1]

    assert first.estimator == "test"
    assert first.parameter_index == 0
    assert first.success_rate == 1.0
    np.testing.assert_allclose(first.bias, 0.0, atol=1e-12)
    np.testing.assert_allclose(first.rmse, 0.1, atol=1e-12)
    np.testing.assert_allclose(
        first.empirical_sd,
        np.sqrt(0.02),
        atol=1e-12,
    )
    assert first.coverage == 1.0
    assert first.mean_objective_evaluations == 4.0
    assert first.mean_demanding_evaluations == 2.0

    assert second.parameter_index == 1
    np.testing.assert_allclose(second.bias, 0.0, atol=1e-12)
    np.testing.assert_allclose(second.rmse, 0.2, atol=1e-12)
    np.testing.assert_allclose(
        second.empirical_sd,
        np.sqrt(0.08),
        atol=1e-12,
    )
    assert second.coverage == 1.0


def test_experiment_respects_estimator_reported_failure() -> None:
    design = LinearIVDesign(n=50, seed=123)

    def runner(data):
        return {
            "theta": data["theta_true"].copy(),
            "standard_errors": np.array([0.1, 0.2]),
            "success": False,
            "message": "optimizer failed",
        }

    results = Experiment(
        design=design,
        estimators={"failed": runner},
        repetitions=2,
        seed=99,
    ).run()

    assert len(results.failed()) == 2
    assert results.n_failures == 2
    assert all(
        record.error == "optimizer failed"
        for record in results.records
    )

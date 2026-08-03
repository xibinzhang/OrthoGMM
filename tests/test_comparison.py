from pathlib import Path
from types import SimpleNamespace

import numpy as np

from orthogmm.simulation import (
    Experiment,
    ExperimentResults,
    LinearIVDesign,
    ReplicationResult,
)


def _record(
    *,
    replication: int,
    estimator: str,
    estimate: np.ndarray,
    covariance: np.ndarray | None,
    objective_value: float | None,
    runtime: float,
    demanding: int | None,
    success: bool = True,
) -> ReplicationResult:
    return ReplicationResult(
        replication=replication,
        estimator=estimator,
        seed=100 + replication,
        true_parameter=np.array([1.0, -0.5]),
        estimate=estimate,
        standard_errors=np.array([0.1, 0.2]) if success else None,
        covariance=covariance,
        objective_value=objective_value,
        comparison_objective_value=objective_value,
        runtime_seconds=runtime,
        objective_evaluations=5,
        demanding_evaluations=demanding,
        success=success,
        error=None if success else "failed",
    )


def _paired_results() -> ExperimentResults:
    identity = np.eye(2)
    return ExperimentResults(
        records=[
            _record(
                replication=0,
                estimator="Full",
                estimate=np.array([1.0, -0.5]),
                covariance=identity,
                objective_value=1.0,
                runtime=2.0,
                demanding=10,
            ),
            _record(
                replication=0,
                estimator="Projection",
                estimate=np.array([1.03, -0.54]),
                covariance=1.5 * identity,
                objective_value=1.25,
                runtime=1.0,
                demanding=2,
            ),
            _record(
                replication=1,
                estimator="Full",
                estimate=np.array([0.98, -0.48]),
                covariance=identity,
                objective_value=0.8,
                runtime=3.0,
                demanding=20,
            ),
            _record(
                replication=1,
                estimator="Projection",
                estimate=np.array([0.98, -0.48]),
                covariance=identity,
                objective_value=0.8,
                runtime=1.5,
                demanding=5,
            ),
        ]
    )


def test_compare_computes_paired_oracle_diagnostics() -> None:
    comparison = _paired_results().compare(
        reference="Full",
        candidate="Projection",
    )
    summary = comparison.summary()

    assert summary.pairs == 2
    assert summary.joint_success_rate == 1.0
    assert summary.convergence_agreement_rate == 1.0
    np.testing.assert_allclose(summary.mean_parameter_distance, 0.025)
    np.testing.assert_allclose(summary.max_parameter_distance, 0.05)
    np.testing.assert_allclose(
        summary.mean_covariance_distance,
        np.sqrt(0.5) / 2.0,
    )
    np.testing.assert_allclose(summary.mean_objective_difference, 0.125)
    np.testing.assert_allclose(summary.mean_absolute_objective_difference, 0.125)
    np.testing.assert_allclose(summary.mean_runtime_speedup, 2.0)
    np.testing.assert_allclose(
        summary.mean_demanding_evaluation_reduction,
        0.775,
    )


def test_compare_exports_summary_and_replications(tmp_path: Path) -> None:
    comparison = _paired_results().compare(
        reference="Full",
        candidate="Projection",
    )

    summary_path = comparison.to_csv(tmp_path / "comparison.csv")
    records_path = comparison.to_csv(
        tmp_path / "replications.csv",
        table="replications",
    )
    latex_path = comparison.to_latex(
        tmp_path / "comparison.tex",
        caption="Oracle comparison",
        label="tab:oracle-comparison",
    )

    assert "mean_parameter_distance" in summary_path.read_text(encoding="utf-8")
    assert "parameter_distance" in records_path.read_text(encoding="utf-8")
    latex = latex_path.read_text(encoding="utf-8")
    assert "\\toprule" in latex
    assert "\\caption{Oracle comparison}" in latex
    assert "Projection" in latex
    assert "Full" in comparison.summary_table()


def test_compare_reports_joint_failure_without_distances() -> None:
    results = ExperimentResults(
        records=[
            _record(
                replication=0,
                estimator="Full",
                estimate=np.array([1.0, -0.5]),
                covariance=np.eye(2),
                objective_value=1.0,
                runtime=2.0,
                demanding=10,
                success=True,
            ),
            _record(
                replication=0,
                estimator="Projection",
                estimate=np.array([]),
                covariance=None,
                objective_value=None,
                runtime=1.0,
                demanding=None,
                success=False,
            ),
        ]
    )

    summary = results.compare(
        reference="Full",
        candidate="Projection",
    ).summary()

    assert summary.joint_success_rate == 0.0
    assert summary.convergence_agreement_rate == 0.0
    assert summary.mean_parameter_distance is None
    assert summary.mean_runtime_speedup is None


def test_experiment_retains_covariance_and_objective_value() -> None:
    design = LinearIVDesign(n=30, seed=123)

    def runner(data):
        return SimpleNamespace(
            theta=data["theta_true"].copy(),
            standard_errors=np.array([0.1, 0.2]),
            covariance=np.diag([0.01, 0.04]),
            objective_value=1.25,
            comparison_objective_value=0.75,
            success=True,
        )

    record = Experiment(
        design=design,
        estimators={"test": runner},
        repetitions=1,
        seed=99,
    ).run().records[0]

    np.testing.assert_allclose(record.covariance, np.diag([0.01, 0.04]))
    assert record.objective_value == 1.25
    assert record.comparison_objective_value == 0.75

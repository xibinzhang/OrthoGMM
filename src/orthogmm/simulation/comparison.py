"""Paired estimator comparisons for Monte Carlo benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from csv import DictWriter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .experiment import ExperimentResults, ReplicationResult


@dataclass(frozen=True)
class ComparisonRecord:
    """Paired diagnostics for two estimators in one replication."""

    replication: int
    seed: int
    reference: str
    candidate: str
    reference_success: bool
    candidate_success: bool
    convergence_agreement: bool
    jointly_successful: bool
    parameter_distance: float | None
    covariance_distance: float | None
    objective_difference: float | None
    objective_absolute_difference: float | None
    runtime_speedup: float | None
    demanding_evaluation_reduction: float | None
    reference_runtime_seconds: float
    candidate_runtime_seconds: float
    reference_demanding_evaluations: int | None
    candidate_demanding_evaluations: int | None


@dataclass(frozen=True)
class ComparisonSummary:
    """Aggregate paired diagnostics for a candidate and reference estimator."""

    reference: str
    candidate: str
    pairs: int
    joint_successes: int
    joint_success_rate: float
    convergence_agreement_rate: float
    mean_parameter_distance: float | None
    median_parameter_distance: float | None
    p95_parameter_distance: float | None
    max_parameter_distance: float | None
    mean_covariance_distance: float | None
    mean_objective_difference: float | None
    mean_absolute_objective_difference: float | None
    mean_runtime_speedup: float | None
    mean_demanding_evaluation_reduction: float | None
    mean_reference_runtime_seconds: float | None
    mean_candidate_runtime_seconds: float | None
    mean_reference_demanding_evaluations: float | None
    mean_candidate_demanding_evaluations: float | None


@dataclass
class EstimatorComparison:
    """Paired comparison of one candidate estimator against a reference."""

    reference: str
    candidate: str
    records: list[ComparisonRecord] = field(default_factory=list)

    @classmethod
    def from_results(
        cls,
        results: ExperimentResults,
        *,
        reference: str,
        candidate: str,
    ) -> EstimatorComparison:
        """Construct a paired comparison from experiment results."""

        if reference == candidate:
            raise ValueError("reference and candidate must be different estimators.")

        available = set(results.estimator_names)
        missing = [name for name in (reference, candidate) if name not in available]
        if missing:
            raise KeyError(
                "Unknown estimator name(s): " + ", ".join(repr(name) for name in missing)
            )

        reference_records = _index_records(results.by_estimator(reference), reference)
        candidate_records = _index_records(results.by_estimator(candidate), candidate)

        if reference_records.keys() != candidate_records.keys():
            reference_only = sorted(reference_records.keys() - candidate_records.keys())
            candidate_only = sorted(candidate_records.keys() - reference_records.keys())
            raise ValueError(
                "Paired comparison requires identical replication/seed keys. "
                f"Reference-only keys: {reference_only}; "
                f"candidate-only keys: {candidate_only}."
            )

        paired_records = [
            _compare_pair(
                reference_records[key],
                candidate_records[key],
                reference=reference,
                candidate=candidate,
            )
            for key in sorted(reference_records)
        ]
        return cls(reference=reference, candidate=candidate, records=paired_records)

    def summarize(self) -> ComparisonSummary:
        """Summarize paired statistical and computational diagnostics."""

        if not self.records:
            raise ValueError("Cannot summarize an empty estimator comparison.")

        joint_records = [record for record in self.records if record.jointly_successful]
        pairs = len(self.records)

        return ComparisonSummary(
            reference=self.reference,
            candidate=self.candidate,
            pairs=pairs,
            joint_successes=len(joint_records),
            joint_success_rate=len(joint_records) / pairs,
            convergence_agreement_rate=float(
                np.mean([record.convergence_agreement for record in self.records])
            ),
            mean_parameter_distance=_mean_metric(joint_records, "parameter_distance"),
            median_parameter_distance=_quantile_metric(
                joint_records, "parameter_distance", 0.5
            ),
            p95_parameter_distance=_quantile_metric(
                joint_records, "parameter_distance", 0.95
            ),
            max_parameter_distance=_max_metric(joint_records, "parameter_distance"),
            mean_covariance_distance=_mean_metric(joint_records, "covariance_distance"),
            mean_objective_difference=_mean_metric(
                joint_records, "objective_difference"
            ),
            mean_absolute_objective_difference=_mean_metric(
                joint_records, "objective_absolute_difference"
            ),
            mean_runtime_speedup=_mean_metric(joint_records, "runtime_speedup"),
            mean_demanding_evaluation_reduction=_mean_metric(
                joint_records, "demanding_evaluation_reduction"
            ),
            mean_reference_runtime_seconds=_mean_metric(
                joint_records, "reference_runtime_seconds"
            ),
            mean_candidate_runtime_seconds=_mean_metric(
                joint_records, "candidate_runtime_seconds"
            ),
            mean_reference_demanding_evaluations=_mean_metric(
                joint_records, "reference_demanding_evaluations"
            ),
            mean_candidate_demanding_evaluations=_mean_metric(
                joint_records, "candidate_demanding_evaluations"
            ),
        )

    def summary(self) -> ComparisonSummary:
        """Alias for :meth:`summarize`."""

        return self.summarize()

    def summary_table(self, *, digits: int = 4) -> str:
        """Return a readable plain-text comparison table."""

        from orthogmm.reporting.comparison import format_comparison_table

        return format_comparison_table(self.summarize(), digits=digits)

    def to_csv(self, path: str | Path, *, table: str = "summary") -> Path:
        """Write summary or replication-level comparison diagnostics to CSV."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        if table == "summary":
            rows: list[dict[str, Any]] = [asdict(self.summarize())]
        elif table == "replications":
            rows = [asdict(record) for record in self.records]
        else:
            raise ValueError("table must be 'summary' or 'replications'.")

        with destination.open("w", newline="", encoding="utf-8") as file:
            writer = DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        return destination

    def to_latex(
        self,
        path: str | Path,
        *,
        digits: int = 4,
        caption: str | None = None,
        label: str | None = None,
    ) -> Path:
        """Write the comparison summary as a booktabs-compatible LaTeX table."""

        from orthogmm.reporting.comparison import write_comparison_latex

        return write_comparison_latex(
            self.summarize(),
            path,
            digits=digits,
            caption=caption,
            label=label,
        )

    def plot_parameter_distance(self, path: str | Path) -> Path:
        """Plot the distribution of paired parameter distances."""

        return self._plot_record_metric(
            path,
            metric="parameter_distance",
            xlabel=r"$\|\widehat{\theta}_{candidate}-\widehat{\theta}_{reference}\|_2$",
        )

    def plot_runtime_speedup(self, path: str | Path) -> Path:
        """Plot the distribution of reference-to-candidate runtime speedups."""

        return self._plot_record_metric(
            path,
            metric="runtime_speedup",
            xlabel="Runtime speedup (reference / candidate)",
        )

    def _plot_record_metric(
        self,
        path: str | Path,
        *,
        metric: str,
        xlabel: str,
    ) -> Path:
        from orthogmm.reporting.comparison import plot_comparison_metric

        return plot_comparison_metric(
            self.records,
            path,
            metric=metric,
            xlabel=xlabel,
        )


def _index_records(
    records: list[ReplicationResult],
    estimator: str,
) -> dict[tuple[int, int], ReplicationResult]:
    indexed: dict[tuple[int, int], ReplicationResult] = {}
    for record in records:
        key = (record.replication, record.seed)
        if key in indexed:
            raise ValueError(
                f"Duplicate {estimator!r} result for replication/seed key {key}."
            )
        indexed[key] = record
    return indexed


def _compare_pair(
    reference_record: ReplicationResult,
    candidate_record: ReplicationResult,
    *,
    reference: str,
    candidate: str,
) -> ComparisonRecord:
    jointly_successful = reference_record.success and candidate_record.success

    parameter_distance: float | None = None
    covariance_distance: float | None = None
    objective_difference: float | None = None
    objective_absolute_difference: float | None = None
    runtime_speedup: float | None = None
    demanding_reduction: float | None = None

    if jointly_successful:
        if reference_record.estimate.shape != candidate_record.estimate.shape:
            raise ValueError(
                "Paired estimates must have the same dimension for "
                f"replication {reference_record.replication}."
            )

        parameter_distance = float(
            np.linalg.norm(candidate_record.estimate - reference_record.estimate)
        )

        if (
            reference_record.covariance is not None
            and candidate_record.covariance is not None
        ):
            if reference_record.covariance.shape != candidate_record.covariance.shape:
                raise ValueError(
                    "Paired covariance matrices must have the same shape for "
                    f"replication {reference_record.replication}."
                )
            covariance_distance = float(
                np.linalg.norm(
                    candidate_record.covariance - reference_record.covariance,
                    ord="fro",
                )
            )

        if (
            reference_record.comparison_objective_value is not None
            and candidate_record.comparison_objective_value is not None
        ):
            objective_difference = float(
                candidate_record.comparison_objective_value
                - reference_record.comparison_objective_value
            )
            objective_absolute_difference = abs(objective_difference)

        if candidate_record.runtime_seconds > 0.0:
            runtime_speedup = float(
                reference_record.runtime_seconds / candidate_record.runtime_seconds
            )

        reference_demanding = reference_record.demanding_evaluations
        candidate_demanding = candidate_record.demanding_evaluations
        if (
            reference_demanding is not None
            and candidate_demanding is not None
            and reference_demanding > 0
        ):
            demanding_reduction = float(
                1.0 - candidate_demanding / reference_demanding
            )

    return ComparisonRecord(
        replication=reference_record.replication,
        seed=reference_record.seed,
        reference=reference,
        candidate=candidate,
        reference_success=reference_record.success,
        candidate_success=candidate_record.success,
        convergence_agreement=(
            reference_record.success == candidate_record.success
        ),
        jointly_successful=jointly_successful,
        parameter_distance=parameter_distance,
        covariance_distance=covariance_distance,
        objective_difference=objective_difference,
        objective_absolute_difference=objective_absolute_difference,
        runtime_speedup=runtime_speedup,
        demanding_evaluation_reduction=demanding_reduction,
        reference_runtime_seconds=reference_record.runtime_seconds,
        candidate_runtime_seconds=candidate_record.runtime_seconds,
        reference_demanding_evaluations=reference_record.demanding_evaluations,
        candidate_demanding_evaluations=candidate_record.demanding_evaluations,
    )


def _finite_metric(records: list[ComparisonRecord], name: str) -> np.ndarray:
    values = [getattr(record, name) for record in records]
    return np.asarray(
        [float(value) for value in values if value is not None and np.isfinite(value)],
        dtype=float,
    )


def _mean_metric(records: list[ComparisonRecord], name: str) -> float | None:
    values = _finite_metric(records, name)
    return float(np.mean(values)) if values.size else None


def _quantile_metric(
    records: list[ComparisonRecord],
    name: str,
    quantile: float,
) -> float | None:
    values = _finite_metric(records, name)
    return float(np.quantile(values, quantile)) if values.size else None


def _max_metric(records: list[ComparisonRecord], name: str) -> float | None:
    values = _finite_metric(records, name)
    return float(np.max(values)) if values.size else None

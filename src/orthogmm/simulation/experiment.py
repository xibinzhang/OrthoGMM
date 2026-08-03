"""
Monte Carlo experiment engine for OrthoGMM.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from csv import DictWriter
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm


FloatArray = NDArray[np.float64]
EstimatorRunner = Callable[[Mapping[str, Any]], Any]


@dataclass(frozen=True)
class ReplicationResult:
    """Results from one estimator in one Monte Carlo replication."""

    replication: int
    estimator: str
    seed: int
    true_parameter: FloatArray
    estimate: FloatArray
    standard_errors: FloatArray | None
    runtime_seconds: float
    objective_evaluations: int | None
    demanding_evaluations: int | None
    success: bool
    error: str | None = None
    covariance: FloatArray | None = None
    objective_value: float | None = None
    comparison_objective_value: float | None = None


@dataclass(frozen=True)
class ParameterSummary:
    """Monte Carlo summary for one estimator and one parameter."""

    estimator: str
    parameter_index: int
    true_value: float
    repetitions: int
    successes: int
    success_rate: float
    bias: float
    rmse: float
    empirical_sd: float
    mean_standard_error: float | None
    coverage: float | None
    mean_runtime_seconds: float
    mean_objective_evaluations: float | None
    mean_demanding_evaluations: float | None


@dataclass
class ExperimentResults:
    """Container for all Monte Carlo replication results."""

    records: list[ReplicationResult] = field(default_factory=list)

    def successful(self) -> list[ReplicationResult]:
        """Return successful estimation records."""

        return [record for record in self.records if record.success]

    def failed(self) -> list[ReplicationResult]:
        """Return failed estimation records."""

        return [record for record in self.records if not record.success]

    def by_estimator(self, name: str) -> list[ReplicationResult]:
        """Return records for one estimator."""

        return [
            record
            for record in self.records
            if record.estimator == name
        ]

    @property
    def n_failures(self) -> int:
        """Number of failed estimator runs."""

        return len(self.failed())

    @property
    def estimator_names(self) -> tuple[str, ...]:
        """Estimator names in first-appearance order."""

        return tuple(dict.fromkeys(record.estimator for record in self.records))


    def compare(
        self,
        *,
        reference: str,
        candidate: str,
    ):
        """Return a paired candidate-versus-reference comparison.

        Estimator results are paired by replication number and simulation seed.
        """

        from .comparison import EstimatorComparison

        return EstimatorComparison.from_results(
            self,
            reference=reference,
            candidate=candidate,
        )

    def summarize(
        self,
        *,
        confidence_level: float = 0.95,
    ) -> list[ParameterSummary]:
        """Compute Monte Carlo summaries by estimator and parameter.

        The summaries include bias, RMSE, empirical standard deviation,
        mean estimated standard error, confidence-interval coverage,
        success rate, runtime, and computational evaluation counts.
        """

        if not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence_level must lie in (0, 1).")

        critical_value = float(
            norm.ppf(0.5 + confidence_level / 2.0)
        )
        summaries: list[ParameterSummary] = []

        for estimator_name in self.estimator_names:
            estimator_records = self.by_estimator(estimator_name)

            if not estimator_records:
                continue

            successful_records = [
                record for record in estimator_records if record.success
            ]

            parameter_dimension = self._parameter_dimension(
                estimator_records
            )

            for parameter_index in range(parameter_dimension):
                true_value = self._true_value(
                    estimator_records,
                    parameter_index,
                )

                if successful_records:
                    estimates = np.asarray(
                        [
                            record.estimate[parameter_index]
                            for record in successful_records
                        ],
                        dtype=float,
                    )
                    errors = estimates - true_value
                    bias = float(np.mean(errors))
                    rmse = float(np.sqrt(np.mean(errors**2)))
                    empirical_sd = float(
                        np.std(estimates, ddof=1)
                        if estimates.size > 1
                        else 0.0
                    )
                    mean_runtime = float(
                        np.mean(
                            [
                                record.runtime_seconds
                                for record in successful_records
                            ]
                        )
                    )
                else:
                    bias = float("nan")
                    rmse = float("nan")
                    empirical_sd = float("nan")
                    mean_runtime = float("nan")

                standard_error_values = [
                    float(record.standard_errors[parameter_index])
                    for record in successful_records
                    if (
                        record.standard_errors is not None
                        and record.standard_errors.size > parameter_index
                        and np.isfinite(
                            record.standard_errors[parameter_index]
                        )
                    )
                ]

                if standard_error_values:
                    mean_standard_error: float | None = float(
                        np.mean(standard_error_values)
                    )

                    covered = []
                    for record in successful_records:
                        if (
                            record.standard_errors is None
                            or record.standard_errors.size <= parameter_index
                        ):
                            continue

                        standard_error = float(
                            record.standard_errors[parameter_index]
                        )
                        estimate = float(
                            record.estimate[parameter_index]
                        )

                        if not (
                            np.isfinite(standard_error)
                            and np.isfinite(estimate)
                        ):
                            continue

                        lower = estimate - critical_value * standard_error
                        upper = estimate + critical_value * standard_error
                        covered.append(lower <= true_value <= upper)

                    coverage: float | None = (
                        float(np.mean(covered)) if covered else None
                    )
                else:
                    mean_standard_error = None
                    coverage = None

                objective_counts = [
                    record.objective_evaluations
                    for record in successful_records
                    if record.objective_evaluations is not None
                ]
                demanding_counts = [
                    record.demanding_evaluations
                    for record in successful_records
                    if record.demanding_evaluations is not None
                ]

                summaries.append(
                    ParameterSummary(
                        estimator=estimator_name,
                        parameter_index=parameter_index,
                        true_value=true_value,
                        repetitions=len(estimator_records),
                        successes=len(successful_records),
                        success_rate=(
                            len(successful_records)
                            / len(estimator_records)
                        ),
                        bias=bias,
                        rmse=rmse,
                        empirical_sd=empirical_sd,
                        mean_standard_error=mean_standard_error,
                        coverage=coverage,
                        mean_runtime_seconds=mean_runtime,
                        mean_objective_evaluations=(
                            float(np.mean(objective_counts))
                            if objective_counts
                            else None
                        ),
                        mean_demanding_evaluations=(
                            float(np.mean(demanding_counts))
                            if demanding_counts
                            else None
                        ),
                    )
                )

        return summaries

    def summary(
        self,
        *,
        confidence_level: float = 0.95,
    ) -> list[ParameterSummary]:
        """Alias for :meth:`summarize` used by the Version 1.0 API."""

        return self.summarize(confidence_level=confidence_level)

    def summary_table(
        self,
        *,
        confidence_level: float = 0.95,
        columns: tuple[str, ...] | None = None,
        digits: int = 4,
    ) -> str:
        """Return a readable plain-text summary table."""

        from orthogmm.reporting.benchmark import (
            _DEFAULT_COLUMNS,
            format_summary_table,
            summary_rows,
        )

        selected_columns = _DEFAULT_COLUMNS if columns is None else columns
        rows = summary_rows(
            self.summarize(confidence_level=confidence_level)
        )
        return format_summary_table(
            rows,
            columns=selected_columns,
            digits=digits,
        )

    def to_latex(
        self,
        path: str | Path,
        *,
        confidence_level: float = 0.95,
        columns: tuple[str, ...] | None = None,
        digits: int = 4,
        caption: str | None = None,
        label: str | None = None,
    ) -> Path:
        """Write the Monte Carlo summary as a LaTeX table."""

        from orthogmm.reporting.benchmark import (
            _DEFAULT_COLUMNS,
            summary_rows,
            write_summary_latex,
        )

        selected_columns = _DEFAULT_COLUMNS if columns is None else columns
        rows = summary_rows(
            self.summarize(confidence_level=confidence_level)
        )
        return write_summary_latex(
            rows,
            path,
            columns=selected_columns,
            digits=digits,
            caption=caption,
            label=label,
        )

    def plot_runtime(self, path: str | Path) -> Path:
        """Plot mean runtime by estimator."""

        return self._plot_summary_metric(
            path,
            metric="mean_runtime_seconds",
            ylabel="Mean runtime (seconds)",
        )

    def plot_rmse(self, path: str | Path) -> Path:
        """Plot RMSE averaged over parameters by estimator."""

        return self._plot_summary_metric(
            path,
            metric="rmse",
            ylabel="RMSE",
        )

    def plot_demanding_evaluations(self, path: str | Path) -> Path:
        """Plot mean demanding evaluations by estimator."""

        return self._plot_summary_metric(
            path,
            metric="mean_demanding_evaluations",
            ylabel="Mean demanding evaluations",
        )

    def _plot_summary_metric(
        self,
        path: str | Path,
        *,
        metric: str,
        ylabel: str,
    ) -> Path:
        from orthogmm.reporting.benchmark import plot_metric, summary_rows

        rows = summary_rows(self.summarize())
        return plot_metric(rows, path, metric=metric, ylabel=ylabel)

    def to_csv(
        self,
        path: str | Path,
        *,
        table: str = "summary",
        confidence_level: float = 0.95,
    ) -> Path:
        """Write replication records or summary statistics to CSV.

        Parameters
        ----------
        path
            Destination file.
        table
            Either ``"summary"`` or ``"replications"``.
        confidence_level
            Confidence level used when ``table="summary"``.
        """

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        if table == "summary":
            rows = [
                asdict(row)
                for row in self.summarize(
                    confidence_level=confidence_level
                )
            ]
        elif table == "replications":
            rows = []
            for record in self.records:
                row = asdict(record)
                row["true_parameter"] = record.true_parameter.tolist()
                row["estimate"] = record.estimate.tolist()
                row["standard_errors"] = (
                    None
                    if record.standard_errors is None
                    else record.standard_errors.tolist()
                )
                row["covariance"] = (
                    None
                    if record.covariance is None
                    else record.covariance.tolist()
                )
                rows.append(row)
        else:
            raise ValueError(
                "table must be 'summary' or 'replications'."
            )

        if not rows:
            raise ValueError("Cannot export an empty experiment result.")

        with destination.open("w", newline="", encoding="utf-8") as file:
            writer = DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        return destination

    @staticmethod
    def _parameter_dimension(
        records: list[ReplicationResult],
    ) -> int:
        for record in records:
            if record.true_parameter.size:
                return int(record.true_parameter.size)

        raise ValueError(
            "Cannot determine the parameter dimension because "
            "no true parameter vector was recorded."
        )

    @staticmethod
    def _true_value(
        records: list[ReplicationResult],
        parameter_index: int,
    ) -> float:
        values = {
            float(record.true_parameter[parameter_index])
            for record in records
            if record.true_parameter.size > parameter_index
        }

        if not values:
            raise ValueError(
                "No true parameter value is available for "
                f"parameter {parameter_index}."
            )

        if len(values) != 1:
            raise ValueError(
                "True parameter values differ across replications "
                f"for parameter {parameter_index}."
            )

        return values.pop()


@dataclass
class Experiment:
    """
    Run a Monte Carlo experiment.

    Parameters
    ----------
    design
        Object with a ``generate(seed=...)`` method.
    estimators
        Mapping from estimator names to callables. Each callable receives
        one simulated-data dictionary and returns an estimation result.
    repetitions
        Number of Monte Carlo replications.
    seed
        Master random-number seed.
    continue_on_error
        Continue the experiment when an estimator fails.
    """

    design: Any
    estimators: Mapping[str, EstimatorRunner]
    repetitions: int = 1000
    seed: int = 12345
    continue_on_error: bool = True

    def __post_init__(self) -> None:
        if self.repetitions <= 0:
            raise ValueError("repetitions must be positive.")

        if not self.estimators:
            raise ValueError("At least one estimator must be supplied.")

        for name, runner in self.estimators.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    "Every estimator must have a non-empty name."
                )

            if not callable(runner):
                raise TypeError(
                    f"Estimator runner {name!r} is not callable."
                )

    def run(self) -> ExperimentResults:
        """Run all Monte Carlo replications."""

        results = ExperimentResults()
        seed_sequence = np.random.SeedSequence(self.seed)
        child_sequences = seed_sequence.spawn(self.repetitions)

        for replication, child_sequence in enumerate(child_sequences):
            replication_seed = int(
                child_sequence.generate_state(
                    1,
                    dtype=np.uint32,
                )[0]
            )

            data = self.design.generate(seed=replication_seed)
            true_parameter = self._extract_true_parameter(data)

            for estimator_name, runner in self.estimators.items():
                record = self._run_estimator(
                    replication=replication,
                    replication_seed=replication_seed,
                    estimator_name=estimator_name,
                    runner=runner,
                    data=data,
                    true_parameter=true_parameter,
                )
                results.records.append(record)

                if not record.success and not self.continue_on_error:
                    raise RuntimeError(
                        f"{estimator_name} failed in replication "
                        f"{replication}: {record.error}"
                    )

        return results

    @staticmethod
    def _run_estimator(
        *,
        replication: int,
        replication_seed: int,
        estimator_name: str,
        runner: EstimatorRunner,
        data: Mapping[str, Any],
        true_parameter: FloatArray,
    ) -> ReplicationResult:
        """Run one estimator and standardise its output."""

        start = perf_counter()

        try:
            result = runner(data)
            elapsed = perf_counter() - start

            estimate = Experiment._extract_array(
                result,
                names=("estimate", "theta", "params"),
                required=True,
            )

            standard_errors = Experiment._extract_array(
                result,
                names=("standard_errors", "std_errors", "se"),
                required=False,
            )
            covariance = Experiment._extract_matrix(
                result,
                names=("covariance", "vcov", "cov"),
            )
            objective_value = Experiment._extract_float(
                result,
                names=("objective_value", "objective", "criterion_value"),
            )
            comparison_objective_value = Experiment._extract_float(
                result,
                names=(
                    "comparison_objective_value",
                    "common_objective_value",
                ),
            )

            if estimate.size != true_parameter.size:
                raise ValueError(
                    "Estimated and true parameter vectors must "
                    "have the same dimension."
                )

            if (
                standard_errors is not None
                and standard_errors.size != estimate.size
            ):
                raise ValueError(
                    "standard_errors must have the same dimension "
                    "as the estimate."
                )

            if covariance is not None and covariance.shape != (estimate.size, estimate.size):
                raise ValueError(
                    "covariance must be square with one row and column "
                    "per estimated parameter."
                )

            objective_evaluations = Experiment._extract_count(
                result,
                top_level_names=(
                    "objective_evaluations",
                    "n_objective_evaluations",
                    "nfev",
                ),
                nested_name="tractable_objective",
            )

            demanding_evaluations = Experiment._extract_count(
                result,
                top_level_names=(
                    "demanding_evaluations",
                    "n_demanding_evaluations",
                ),
                nested_name="demanding_moments_total",
            )

            result_success = Experiment._extract_success(result)

            if result_success:
                error = None
            else:
                message = Experiment._extract_message(result)
                error = message or "Estimator reported failure."

            return ReplicationResult(
                replication=replication,
                estimator=estimator_name,
                seed=replication_seed,
                true_parameter=true_parameter.copy(),
                estimate=estimate,
                standard_errors=standard_errors,
                covariance=covariance,
                objective_value=objective_value,
                comparison_objective_value=comparison_objective_value,
                runtime_seconds=elapsed,
                objective_evaluations=objective_evaluations,
                demanding_evaluations=demanding_evaluations,
                success=result_success,
                error=error,
            )

        except Exception as error:
            elapsed = perf_counter() - start

            return ReplicationResult(
                replication=replication,
                estimator=estimator_name,
                seed=replication_seed,
                true_parameter=true_parameter.copy(),
                estimate=np.asarray([], dtype=float),
                standard_errors=None,
                covariance=None,
                objective_value=None,
                comparison_objective_value=None,
                runtime_seconds=elapsed,
                objective_evaluations=None,
                demanding_evaluations=None,
                success=False,
                error=f"{type(error).__name__}: {error}",
            )

    @staticmethod
    def _extract_true_parameter(
        data: Mapping[str, Any],
    ) -> FloatArray:
        if "theta_true" not in data:
            raise KeyError(
                "Simulation data must contain 'theta_true'."
            )

        true_parameter = np.asarray(
            data["theta_true"],
            dtype=float,
        )

        if true_parameter.ndim != 1:
            raise ValueError(
                "theta_true must be one-dimensional."
            )

        if not np.all(np.isfinite(true_parameter)):
            raise ValueError(
                "theta_true contains non-finite values."
            )

        return true_parameter

    @staticmethod
    def _extract_array(
        result: Any,
        *,
        names: tuple[str, ...],
        required: bool,
    ) -> FloatArray | None:
        """Extract an array from either an object or a dictionary."""

        value: Any = None
        found = False

        for name in names:
            if isinstance(result, Mapping) and name in result:
                value = result[name]
                found = True
                break

            if hasattr(result, name):
                value = getattr(result, name)
                found = True
                break

        if not found:
            if required:
                joined = ", ".join(names)
                raise AttributeError(
                    f"Result does not contain any of: {joined}."
                )
            return None

        array = np.asarray(value, dtype=float)

        if array.ndim != 1:
            raise ValueError(
                "Estimated parameters and standard errors "
                "must be one-dimensional."
            )

        if not np.all(np.isfinite(array)):
            raise ValueError(
                "Extracted array contains non-finite values."
            )

        return array

    @staticmethod
    def _extract_matrix(
        result: Any,
        *,
        names: tuple[str, ...],
    ) -> FloatArray | None:
        """Extract a finite two-dimensional matrix from a result."""

        value: Any = None
        found = False
        for name in names:
            if isinstance(result, Mapping) and name in result:
                value = result[name]
                found = True
                break
            if hasattr(result, name):
                value = getattr(result, name)
                found = True
                break

        if not found or value is None:
            return None

        matrix = np.asarray(value, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("Extracted covariance matrix must be two-dimensional.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Extracted covariance matrix contains non-finite values.")
        return matrix

    @staticmethod
    def _extract_float(
        result: Any,
        *,
        names: tuple[str, ...],
    ) -> float | None:
        """Extract one finite scalar from a result."""

        for name in names:
            if isinstance(result, Mapping) and name in result:
                value = result[name]
                break
            if hasattr(result, name):
                value = getattr(result, name)
                break
        else:
            return None

        if value is None:
            return None
        scalar = float(value)
        if not np.isfinite(scalar):
            raise ValueError("Extracted scalar contains a non-finite value.")
        return scalar

    @staticmethod
    def _extract_count(
        result: Any,
        *,
        top_level_names: tuple[str, ...],
        nested_name: str,
    ) -> int | None:
        """Extract a computational count from generic or OrthoGMM results."""

        for name in top_level_names:
            if isinstance(result, Mapping) and name in result:
                value = result[name]
                return None if value is None else int(value)

            if hasattr(result, name):
                value = getattr(result, name)
                return None if value is None else int(value)

        counts = (
            result.get("counts")
            if isinstance(result, Mapping)
            else getattr(result, "counts", None)
        )

        if counts is None:
            return None

        if isinstance(counts, Mapping) and nested_name in counts:
            value = counts[nested_name]
            return None if value is None else int(value)

        if hasattr(counts, nested_name):
            value = getattr(counts, nested_name)
            return None if value is None else int(value)

        return None

    @staticmethod
    def _extract_success(result: Any) -> bool:
        if isinstance(result, Mapping) and "success" in result:
            return bool(result["success"])

        if hasattr(result, "success"):
            return bool(getattr(result, "success"))

        return True

    @staticmethod
    def _extract_message(result: Any) -> str | None:
        if isinstance(result, Mapping) and "message" in result:
            value = result["message"]
            return None if value is None else str(value)

        if hasattr(result, "message"):
            value = getattr(result, "message")
            return None if value is None else str(value)

        return None


@dataclass
class MonteCarloBenchmark(Experiment):
    """Version 1.0 benchmark entry point.

    This class intentionally extends :class:`Experiment`: existing scripts can
    keep using ``Experiment``, while new user code can use the more descriptive
    benchmark name without any behavioural differences.
    """

    pass

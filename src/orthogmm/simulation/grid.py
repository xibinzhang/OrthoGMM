"""Multi-design Monte Carlo grid benchmarks for OrthoGMM."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from csv import DictWriter
from itertools import product
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .comparison import EstimatorComparison
from .experiment import (
    EstimatorRunner,
    ExperimentResults,
    MonteCarloBenchmark,
    ParameterSummary,
)


DesignFactory = Callable[..., Any]


@dataclass(frozen=True)
class GridCellResult:
    """Results for one design-parameter combination in a grid benchmark."""

    index: int
    parameters: Mapping[str, Any]
    seed: int
    results: ExperimentResults

    @property
    def label(self) -> str:
        """Human-readable design-cell label."""

        return ", ".join(
            f"{name}={value}" for name, value in self.parameters.items()
        )


@dataclass(frozen=True)
class GridParameterSummary:
    """One parameter summary augmented with its design-cell metadata."""

    cell_index: int
    cell_seed: int
    design_parameters: Mapping[str, Any]
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

    @classmethod
    def from_parameter_summary(
        cls,
        cell: GridCellResult,
        summary: ParameterSummary,
    ) -> GridParameterSummary:
        """Attach design metadata to a single-cell parameter summary."""

        return cls(
            cell_index=cell.index,
            cell_seed=cell.seed,
            design_parameters=dict(cell.parameters),
            estimator=summary.estimator,
            parameter_index=summary.parameter_index,
            true_value=summary.true_value,
            repetitions=summary.repetitions,
            successes=summary.successes,
            success_rate=summary.success_rate,
            bias=summary.bias,
            rmse=summary.rmse,
            empirical_sd=summary.empirical_sd,
            mean_standard_error=summary.mean_standard_error,
            coverage=summary.coverage,
            mean_runtime_seconds=summary.mean_runtime_seconds,
            mean_objective_evaluations=summary.mean_objective_evaluations,
            mean_demanding_evaluations=summary.mean_demanding_evaluations,
        )

    def as_row(self, grid_names: Sequence[str]) -> dict[str, Any]:
        """Return a flat row suitable for tables and CSV export."""

        row: dict[str, Any] = {
            "cell_index": self.cell_index,
            "cell_seed": self.cell_seed,
        }
        for name in grid_names:
            row[name] = self.design_parameters.get(name)
        row.update(
            {
                "estimator": self.estimator,
                "parameter_index": self.parameter_index,
                "true_value": self.true_value,
                "repetitions": self.repetitions,
                "successes": self.successes,
                "success_rate": self.success_rate,
                "bias": self.bias,
                "rmse": self.rmse,
                "empirical_sd": self.empirical_sd,
                "mean_standard_error": self.mean_standard_error,
                "coverage": self.coverage,
                "mean_runtime_seconds": self.mean_runtime_seconds,
                "mean_objective_evaluations": (
                    self.mean_objective_evaluations
                ),
                "mean_demanding_evaluations": (
                    self.mean_demanding_evaluations
                ),
            }
        )
        return row


@dataclass(frozen=True)
class GridComparisonCell:
    """Paired estimator comparison for one design cell."""

    cell_index: int
    cell_seed: int
    design_parameters: Mapping[str, Any]
    comparison: EstimatorComparison

    def as_summary_row(self, grid_names: Sequence[str]) -> dict[str, Any]:
        """Return a flat paired-comparison summary row."""

        row: dict[str, Any] = {
            "cell_index": self.cell_index,
            "cell_seed": self.cell_seed,
        }
        for name in grid_names:
            row[name] = self.design_parameters.get(name)
        row.update(asdict(self.comparison.summarize()))
        return row


@dataclass
class GridEstimatorComparison:
    """Paired candidate-versus-reference diagnostics across a design grid."""

    reference: str
    candidate: str
    grid_names: tuple[str, ...]
    cells: list[GridComparisonCell] = field(default_factory=list)

    def summary_rows(self) -> list[dict[str, Any]]:
        """Return one flattened paired-comparison row per grid cell."""

        return [
            cell.as_summary_row(self.grid_names)
            for cell in self.cells
        ]

    def summary_table(
        self,
        *,
        columns: Sequence[str] | None = None,
        digits: int = 4,
    ) -> str:
        """Return a readable grid-comparison table."""

        from orthogmm.reporting.grid import format_grid_table

        selected = (
            tuple(self.grid_names)
            + (
                "candidate",
                "reference",
                "joint_success_rate",
                "mean_parameter_distance",
                "p95_parameter_distance",
                "mean_covariance_distance",
                "mean_runtime_speedup",
                "mean_demanding_evaluation_reduction",
            )
            if columns is None
            else tuple(columns)
        )
        return format_grid_table(
            self.summary_rows(),
            columns=selected,
            digits=digits,
        )

    def to_csv(self, path: str | Path, *, table: str = "summary") -> Path:
        """Write cell summaries or paired replication diagnostics to CSV."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        if table == "summary":
            rows = self.summary_rows()
        elif table == "replications":
            rows = []
            for cell in self.cells:
                prefix: dict[str, Any] = {
                    "cell_index": cell.cell_index,
                    "cell_seed": cell.cell_seed,
                }
                prefix.update(
                    {
                        name: cell.design_parameters.get(name)
                        for name in self.grid_names
                    }
                )
                for record in cell.comparison.records:
                    row = dict(prefix)
                    row.update(asdict(record))
                    rows.append(row)
        else:
            raise ValueError("table must be 'summary' or 'replications'.")

        _write_rows(destination, rows)
        return destination

    def to_latex(
        self,
        path: str | Path,
        *,
        columns: Sequence[str] | None = None,
        digits: int = 4,
        caption: str | None = None,
        label: str | None = None,
    ) -> Path:
        """Write cell-level paired-comparison summaries to LaTeX."""

        from orthogmm.reporting.grid import write_grid_latex

        selected = (
            tuple(self.grid_names)
            + (
                "candidate",
                "reference",
                "joint_success_rate",
                "mean_parameter_distance",
                "p95_parameter_distance",
                "mean_runtime_speedup",
                "mean_demanding_evaluation_reduction",
            )
            if columns is None
            else tuple(columns)
        )
        return write_grid_latex(
            self.summary_rows(),
            path,
            columns=selected,
            digits=digits,
            caption=caption,
            label=label,
        )

    def plot_metric(
        self,
        path: str | Path,
        *,
        x: str,
        metric: str,
        ylabel: str | None = None,
    ) -> Path:
        """Plot a paired-comparison metric against one grid dimension."""

        from orthogmm.reporting.grid import plot_grid_metric

        if x not in self.grid_names:
            raise KeyError(f"Unknown grid parameter {x!r}.")
        return plot_grid_metric(
            self.summary_rows(),
            path,
            x=x,
            metric=metric,
            series_columns=tuple(
                name for name in self.grid_names if name != x
            ),
            ylabel=ylabel,
        )

    def plot_parameter_distance(self, path: str | Path, *, x: str) -> Path:
        """Plot mean paired parameter distance over a grid dimension."""

        return self.plot_metric(
            path,
            x=x,
            metric="mean_parameter_distance",
            ylabel="Mean parameter distance",
        )

    def plot_runtime_speedup(self, path: str | Path, *, x: str) -> Path:
        """Plot mean reference-to-candidate runtime speedup."""

        return self.plot_metric(
            path,
            x=x,
            metric="mean_runtime_speedup",
            ylabel="Runtime speedup (reference / candidate)",
        )


@dataclass
class GridResults:
    """Results from a collection of Monte Carlo design cells."""

    grid_names: tuple[str, ...]
    cells: list[GridCellResult] = field(default_factory=list)

    @property
    def n_cells(self) -> int:
        """Number of completed design cells."""

        return len(self.cells)

    def by_parameters(self, **parameters: Any) -> GridCellResult:
        """Return the unique cell matching the supplied design parameters."""

        if not parameters:
            raise ValueError("At least one design parameter must be supplied.")
        unknown = set(parameters) - set(self.grid_names)
        if unknown:
            raise KeyError(
                "Unknown grid parameter(s): "
                + ", ".join(sorted(unknown))
            )
        matches = [
            cell
            for cell in self.cells
            if all(cell.parameters.get(name) == value for name, value in parameters.items())
        ]
        if not matches:
            raise KeyError(f"No grid cell matches {parameters!r}.")
        if len(matches) > 1:
            raise ValueError(
                f"Parameters {parameters!r} identify more than one grid cell."
            )
        return matches[0]

    def summarize(
        self,
        *,
        confidence_level: float = 0.95,
    ) -> list[GridParameterSummary]:
        """Summarize every estimator and parameter in every grid cell."""

        summaries: list[GridParameterSummary] = []
        for cell in self.cells:
            summaries.extend(
                GridParameterSummary.from_parameter_summary(cell, summary)
                for summary in cell.results.summarize(
                    confidence_level=confidence_level
                )
            )
        return summaries

    def summary(
        self,
        *,
        confidence_level: float = 0.95,
    ) -> list[GridParameterSummary]:
        """Alias for :meth:`summarize`."""

        return self.summarize(confidence_level=confidence_level)

    def summary_rows(
        self,
        *,
        confidence_level: float = 0.95,
    ) -> list[dict[str, Any]]:
        """Return flattened grid summary rows."""

        return [
            summary.as_row(self.grid_names)
            for summary in self.summarize(
                confidence_level=confidence_level
            )
        ]

    def summary_table(
        self,
        *,
        confidence_level: float = 0.95,
        columns: Sequence[str] | None = None,
        digits: int = 4,
    ) -> str:
        """Return a readable multi-design summary table."""

        from orthogmm.reporting.grid import format_grid_table

        selected = (
            tuple(self.grid_names)
            + (
                "estimator",
                "parameter_index",
                "bias",
                "rmse",
                "coverage",
                "mean_runtime_seconds",
                "mean_demanding_evaluations",
            )
            if columns is None
            else tuple(columns)
        )
        return format_grid_table(
            self.summary_rows(confidence_level=confidence_level),
            columns=selected,
            digits=digits,
        )

    def to_csv(
        self,
        path: str | Path,
        *,
        table: str = "summary",
        confidence_level: float = 0.95,
    ) -> Path:
        """Write aggregated summaries or all replication records to CSV."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        if table == "summary":
            rows = self.summary_rows(confidence_level=confidence_level)
        elif table == "replications":
            rows = []
            for cell in self.cells:
                prefix: dict[str, Any] = {
                    "cell_index": cell.index,
                    "cell_seed": cell.seed,
                }
                prefix.update(
                    {
                        name: cell.parameters.get(name)
                        for name in self.grid_names
                    }
                )
                for record in cell.results.records:
                    row = dict(prefix)
                    payload = asdict(record)
                    payload["true_parameter"] = record.true_parameter.tolist()
                    payload["estimate"] = record.estimate.tolist()
                    payload["standard_errors"] = (
                        None
                        if record.standard_errors is None
                        else record.standard_errors.tolist()
                    )
                    payload["covariance"] = (
                        None
                        if record.covariance is None
                        else record.covariance.tolist()
                    )
                    row.update(payload)
                    rows.append(row)
        else:
            raise ValueError("table must be 'summary' or 'replications'.")

        _write_rows(destination, rows)
        return destination

    def to_latex(
        self,
        path: str | Path,
        *,
        confidence_level: float = 0.95,
        columns: Sequence[str] | None = None,
        digits: int = 4,
        caption: str | None = None,
        label: str | None = None,
    ) -> Path:
        """Write the aggregated grid summary as a LaTeX table."""

        from orthogmm.reporting.grid import write_grid_latex

        selected = (
            tuple(self.grid_names)
            + (
                "estimator",
                "parameter_index",
                "bias",
                "rmse",
                "coverage",
                "mean_runtime_seconds",
                "mean_demanding_evaluations",
            )
            if columns is None
            else tuple(columns)
        )
        return write_grid_latex(
            self.summary_rows(confidence_level=confidence_level),
            path,
            columns=selected,
            digits=digits,
            caption=caption,
            label=label,
        )

    def compare(
        self,
        *,
        reference: str,
        candidate: str,
    ) -> GridEstimatorComparison:
        """Compute paired candidate-versus-reference diagnostics by cell."""

        comparison_cells = [
            GridComparisonCell(
                cell_index=cell.index,
                cell_seed=cell.seed,
                design_parameters=dict(cell.parameters),
                comparison=cell.results.compare(
                    reference=reference,
                    candidate=candidate,
                ),
            )
            for cell in self.cells
        ]
        return GridEstimatorComparison(
            reference=reference,
            candidate=candidate,
            grid_names=self.grid_names,
            cells=comparison_cells,
        )

    def plot_metric(
        self,
        path: str | Path,
        *,
        x: str,
        metric: str,
        estimator: str | None = None,
        parameter_index: int | None = None,
        ylabel: str | None = None,
    ) -> Path:
        """Plot a summary metric against one grid dimension."""

        from orthogmm.reporting.grid import plot_grid_metric

        if x not in self.grid_names:
            raise KeyError(f"Unknown grid parameter {x!r}.")

        rows = self.summary_rows()
        if estimator is not None:
            rows = [row for row in rows if row["estimator"] == estimator]
        if parameter_index is not None:
            rows = [
                row
                for row in rows
                if row["parameter_index"] == parameter_index
            ]

        series_columns: list[str] = []
        if estimator is None:
            series_columns.append("estimator")
        if parameter_index is None:
            series_columns.append("parameter_index")
        series_columns.extend(name for name in self.grid_names if name != x)

        return plot_grid_metric(
            rows,
            path,
            x=x,
            metric=metric,
            series_columns=tuple(series_columns),
            ylabel=ylabel,
        )

    def plot_runtime(self, path: str | Path, *, x: str) -> Path:
        """Plot mean runtime against one grid dimension by estimator."""

        return self.plot_metric(
            path,
            x=x,
            metric="mean_runtime_seconds",
            parameter_index=0,
            ylabel="Mean runtime (seconds)",
        )

    def plot_rmse(
        self,
        path: str | Path,
        *,
        x: str,
        parameter_index: int = 0,
    ) -> Path:
        """Plot RMSE against one grid dimension by estimator."""

        return self.plot_metric(
            path,
            x=x,
            metric="rmse",
            parameter_index=parameter_index,
            ylabel="RMSE",
        )

    def plot_demanding_evaluations(
        self,
        path: str | Path,
        *,
        x: str,
    ) -> Path:
        """Plot mean demanding evaluations against one grid dimension."""

        return self.plot_metric(
            path,
            x=x,
            metric="mean_demanding_evaluations",
            parameter_index=0,
            ylabel="Mean demanding evaluations",
        )


@dataclass
class GridBenchmark:
    """Run a Monte Carlo benchmark over a Cartesian design grid.

    Parameters
    ----------
    design_factory
        Callable, typically a design class, accepting design parameters as
        keyword arguments and returning an object with ``generate(seed=...)``.
    grid
        Mapping from varied design-parameter names to non-empty sequences.
        The Cartesian product of these sequences defines the design cells.
    estimators
        Mapping from estimator names to single-replication runner callables.
    repetitions
        Monte Carlo replications in each design cell.
    seed
        Master seed used to derive deterministic cell-level seeds.
    base_design_parameters
        Fixed keyword arguments supplied to every design instance.
    continue_on_error
        Passed to each underlying :class:`MonteCarloBenchmark`.
    """

    design_factory: DesignFactory
    grid: Mapping[str, Sequence[Any]]
    estimators: Mapping[str, EstimatorRunner]
    repetitions: int = 1000
    seed: int = 12345
    base_design_parameters: Mapping[str, Any] = field(default_factory=dict)
    continue_on_error: bool = True

    def __post_init__(self) -> None:
        if not callable(self.design_factory):
            raise TypeError("design_factory must be callable.")
        if self.repetitions <= 0:
            raise ValueError("repetitions must be positive.")
        if not self.grid:
            raise ValueError("grid must contain at least one parameter.")
        if not self.estimators:
            raise ValueError("At least one estimator must be supplied.")

        normalized: dict[str, tuple[Any, ...]] = {}
        for name, values in self.grid.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    "Every grid parameter must have a non-empty string name."
                )
            if name in self.base_design_parameters:
                raise ValueError(
                    f"Grid parameter {name!r} is also fixed in "
                    "base_design_parameters."
                )
            if isinstance(values, (str, bytes)):
                raise TypeError(
                    f"Grid values for {name!r} must be a sequence, "
                    "not a string."
                )
            try:
                materialized = tuple(values)
            except TypeError as error:
                raise TypeError(
                    f"Grid values for {name!r} must be iterable."
                ) from error
            if not materialized:
                raise ValueError(
                    f"Grid values for {name!r} cannot be empty."
                )
            normalized[name] = materialized

        self.grid = normalized
        self.base_design_parameters = dict(self.base_design_parameters)

    @property
    def grid_names(self) -> tuple[str, ...]:
        """Grid parameter names in insertion order."""

        return tuple(self.grid)

    @property
    def n_cells(self) -> int:
        """Number of Cartesian design cells."""

        count = 1
        for values in self.grid.values():
            count *= len(values)
        return count

    def parameter_combinations(self) -> list[dict[str, Any]]:
        """Return all design-cell parameter combinations in run order."""

        names = self.grid_names
        return [
            dict(zip(names, values))
            for values in product(*(self.grid[name] for name in names))
        ]

    def run(self) -> GridResults:
        """Run all grid cells and aggregate their experiment results."""

        combinations = self.parameter_combinations()
        seed_sequence = np.random.SeedSequence(self.seed)
        child_sequences = seed_sequence.spawn(len(combinations))
        cells: list[GridCellResult] = []

        for index, (parameters, child_sequence) in enumerate(
            zip(combinations, child_sequences)
        ):
            cell_seed = int(
                child_sequence.generate_state(1, dtype=np.uint32)[0]
            )
            design_parameters = dict(self.base_design_parameters)
            design_parameters.update(parameters)
            try:
                design = self.design_factory(**design_parameters)
            except Exception as error:
                raise RuntimeError(
                    "Failed to construct grid design for cell "
                    f"{index} with parameters {design_parameters!r}: "
                    f"{type(error).__name__}: {error}"
                ) from error

            benchmark = MonteCarloBenchmark(
                design=design,
                estimators=self.estimators,
                repetitions=self.repetitions,
                seed=cell_seed,
                continue_on_error=self.continue_on_error,
            )
            results = benchmark.run()
            cells.append(
                GridCellResult(
                    index=index,
                    parameters=design_parameters,
                    seed=cell_seed,
                    results=results,
                )
            )

        all_names = tuple(
            dict.fromkeys(
                (*self.base_design_parameters.keys(), *self.grid_names)
            )
        )
        return GridResults(grid_names=all_names, cells=cells)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot export an empty grid result.")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

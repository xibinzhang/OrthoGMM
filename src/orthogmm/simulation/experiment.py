"""
Monte Carlo experiment engine for OrthoGMM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Mapping

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
EstimatorRunner = Callable[[Mapping[str, Any]], Any]


@dataclass(frozen=True)
class ReplicationResult:
    """Results from one estimator in one Monte Carlo replication."""

    replication: int
    estimator: str
    seed: int
    estimate: FloatArray
    standard_errors: FloatArray | None
    runtime_seconds: float
    objective_evaluations: int | None
    demanding_evaluations: int | None
    success: bool
    error: str | None = None


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
                child_sequence.generate_state(1, dtype=np.uint32)[0]
            )

            data = self.design.generate(seed=replication_seed)

            for estimator_name, runner in self.estimators.items():
                record = self._run_estimator(
                    replication=replication,
                    replication_seed=replication_seed,
                    estimator_name=estimator_name,
                    runner=runner,
                    data=data,
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

            objective_evaluations = Experiment._extract_integer(
                result,
                names=(
                    "objective_evaluations",
                    "n_objective_evaluations",
                    "nfev",
                ),
            )

            demanding_evaluations = Experiment._extract_integer(
                result,
                names=(
                    "demanding_evaluations",
                    "n_demanding_evaluations",
                ),
            )

            return ReplicationResult(
                replication=replication,
                estimator=estimator_name,
                seed=replication_seed,
                estimate=estimate,
                standard_errors=standard_errors,
                runtime_seconds=elapsed,
                objective_evaluations=objective_evaluations,
                demanding_evaluations=demanding_evaluations,
                success=True,
            )

        except Exception as error:
            elapsed = perf_counter() - start

            return ReplicationResult(
                replication=replication,
                estimator=estimator_name,
                seed=replication_seed,
                estimate=np.asarray([], dtype=float),
                standard_errors=None,
                runtime_seconds=elapsed,
                objective_evaluations=None,
                demanding_evaluations=None,
                success=False,
                error=f"{type(error).__name__}: {error}",
            )

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
            raise ValueError("Estimated parameters must be one-dimensional.")

        return array

    @staticmethod
    def _extract_integer(
        result: Any,
        *,
        names: tuple[str, ...],
    ) -> int | None:
        """Extract an optional integer diagnostic."""

        for name in names:
            if isinstance(result, Mapping) and name in result:
                value = result[name]
                return None if value is None else int(value)

            if hasattr(result, name):
                value = getattr(result, name)
                return None if value is None else int(value)

        return None
    
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from orthogmm import GridBenchmark


@dataclass
class WorkloadDesign:
    """Design whose workload parameter does not alter simulated data."""

    workload: int

    def generate(self, seed=None):
        rng = np.random.default_rng(seed)
        return {
            "theta_true": np.asarray([0.0], dtype=float),
            "noise": float(rng.normal()),
            "workload": self.workload,
        }


def estimator(data):
    return {
        "estimate": np.asarray([data["noise"]], dtype=float),
        "standard_errors": np.asarray([1.0], dtype=float),
        "covariance": np.asarray([[1.0]], dtype=float),
        "success": True,
        "objective_evaluations": 1,
        "demanding_evaluations": 1,
    }


def build_grid(*, common_random_numbers: bool) -> GridBenchmark:
    return GridBenchmark(
        design_factory=WorkloadDesign,
        grid={"workload": [10, 20, 40]},
        estimators={"Estimator": estimator},
        repetitions=5,
        seed=321,
        common_random_numbers=common_random_numbers,
    )


def test_default_uses_distinct_deterministic_cell_seeds():
    benchmark = build_grid(common_random_numbers=False)

    first = benchmark.cell_seeds()
    second = benchmark.cell_seeds()

    assert first == second
    assert len(first) == benchmark.n_cells
    assert len(set(first)) == benchmark.n_cells


def test_common_random_numbers_reuse_cell_master_seed():
    benchmark = build_grid(common_random_numbers=True)

    assert benchmark.cell_seeds() == [321, 321, 321]


def test_common_random_numbers_reuse_replication_samples_across_cells():
    results = build_grid(common_random_numbers=True).run()

    first_cell = results.cells[0].results.by_estimator("Estimator")
    first_seeds = [record.seed for record in first_cell]
    first_estimates = [record.estimate.tolist() for record in first_cell]

    for cell in results.cells[1:]:
        records = cell.results.by_estimator("Estimator")
        assert [record.seed for record in records] == first_seeds
        assert [record.estimate.tolist() for record in records] == first_estimates


def test_independent_cells_do_not_reuse_replication_seed_sequences():
    results = build_grid(common_random_numbers=False).run()

    seed_sequences = [
        [record.seed for record in cell.results.by_estimator("Estimator")]
        for cell in results.cells
    ]

    assert len({tuple(seeds) for seeds in seed_sequences}) == results.n_cells


def test_common_random_numbers_must_be_boolean():
    with pytest.raises(TypeError, match="must be a boolean"):
        GridBenchmark(
            design_factory=WorkloadDesign,
            grid={"workload": [10]},
            estimators={"Estimator": estimator},
            common_random_numbers="yes",  # type: ignore[arg-type]
        )

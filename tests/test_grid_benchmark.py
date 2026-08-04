from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from orthogmm import GridBenchmark
from orthogmm.simulation import GridResults


@dataclass
class TinyDesign:
    n: int
    scale: float = 1.0

    def generate(self, seed=None):
        rng = np.random.default_rng(seed)
        return {
            "theta_true": np.asarray([self.scale], dtype=float),
            "noise": float(rng.normal() / np.sqrt(self.n)),
        }


def reference_runner(data):
    truth = np.asarray(data["theta_true"], dtype=float)
    return {
        "estimate": truth,
        "standard_errors": np.asarray([0.1], dtype=float),
        "covariance": np.asarray([[0.01]], dtype=float),
        "success": True,
        "objective_evaluations": 12,
        "demanding_evaluations": 40,
    }


def candidate_runner(data):
    truth = np.asarray(data["theta_true"], dtype=float)
    estimate = truth + np.asarray([0.1 * data["noise"]], dtype=float)
    return {
        "estimate": estimate,
        "standard_errors": np.asarray([0.1], dtype=float),
        "covariance": np.asarray([[0.011]], dtype=float),
        "success": True,
        "objective_evaluations": 8,
        "demanding_evaluations": 5,
    }


def build_grid():
    return GridBenchmark(
        design_factory=TinyDesign,
        grid={"n": [25, 100], "scale": [1.0, 2.0]},
        estimators={
            "Candidate": candidate_runner,
            "Reference": reference_runner,
        },
        repetitions=4,
        seed=321,
    )


def test_grid_benchmark_cartesian_product_and_lookup():
    benchmark = build_grid()

    assert benchmark.grid_names == ("n", "scale")
    assert benchmark.n_cells == 4
    assert benchmark.parameter_combinations() == [
        {"n": 25, "scale": 1.0},
        {"n": 25, "scale": 2.0},
        {"n": 100, "scale": 1.0},
        {"n": 100, "scale": 2.0},
    ]

    results = benchmark.run()
    assert isinstance(results, GridResults)
    assert results.n_cells == 4
    assert len(results.cells[0].results.records) == 8
    assert results.by_parameters(n=100, scale=2.0).parameters == {
        "n": 100,
        "scale": 2.0,
    }


def test_grid_benchmark_is_reproducible():
    first = build_grid().run()
    second = build_grid().run()

    assert [cell.seed for cell in first.cells] == [
        cell.seed for cell in second.cells
    ]
    first_estimates = [
        record.estimate.tolist()
        for cell in first.cells
        for record in cell.results.records
    ]
    second_estimates = [
        record.estimate.tolist()
        for cell in second.cells
        for record in cell.results.records
    ]
    assert first_estimates == second_estimates


def test_grid_summary_and_paired_comparison():
    results = build_grid().run()
    summaries = results.summary()

    assert len(summaries) == 8
    assert {summary.estimator for summary in summaries} == {
        "Candidate",
        "Reference",
    }
    assert all(summary.repetitions == 4 for summary in summaries)

    comparison = results.compare(
        reference="Reference",
        candidate="Candidate",
    )
    rows = comparison.summary_rows()

    assert len(rows) == 4
    assert all(row["joint_success_rate"] == 1.0 for row in rows)
    assert all(
        row["mean_demanding_evaluation_reduction"] == pytest.approx(0.875)
        for row in rows
    )
    assert all(row["mean_parameter_distance"] is not None for row in rows)


def test_grid_exports_and_tables(tmp_path: Path):
    results = build_grid().run()
    comparison = results.compare(
        reference="Reference",
        candidate="Candidate",
    )

    summary_csv = results.to_csv(tmp_path / "grid_summary.csv")
    replications_csv = results.to_csv(
        tmp_path / "grid_replications.csv",
        table="replications",
    )
    summary_tex = results.to_latex(tmp_path / "grid_summary.tex")
    comparison_csv = comparison.to_csv(tmp_path / "comparison.csv")
    comparison_tex = comparison.to_latex(tmp_path / "comparison.tex")

    assert "n,scale" in summary_csv.read_text(encoding="utf-8").splitlines()[0]
    assert "true_parameter" in replications_csv.read_text(encoding="utf-8")
    assert "\\begin{table}" in summary_tex.read_text(encoding="utf-8")
    assert "mean_parameter_distance" in comparison_csv.read_text(encoding="utf-8")
    assert "\\begin{table}" in comparison_tex.read_text(encoding="utf-8")
    assert "Candidate" in results.summary_table()
    assert "Candidate" in comparison.summary_table()


def test_grid_plots(tmp_path: Path):
    pytest.importorskip("matplotlib")
    results = build_grid().run()
    comparison = results.compare(
        reference="Reference",
        candidate="Candidate",
    )

    assert results.plot_runtime(tmp_path / "runtime.pdf", x="n").exists()
    assert results.plot_rmse(tmp_path / "rmse.pdf", x="n").exists()
    assert results.plot_demanding_evaluations(
        tmp_path / "demanding.pdf",
        x="n",
    ).exists()
    assert comparison.plot_parameter_distance(
        tmp_path / "distance.pdf",
        x="n",
    ).exists()
    assert comparison.plot_runtime_speedup(
        tmp_path / "speedup.pdf",
        x="n",
    ).exists()


def test_grid_validation():
    with pytest.raises(ValueError, match="at least one parameter"):
        GridBenchmark(
            design_factory=TinyDesign,
            grid={},
            estimators={"Reference": reference_runner},
        )

    with pytest.raises(ValueError, match="cannot be empty"):
        GridBenchmark(
            design_factory=TinyDesign,
            grid={"n": []},
            estimators={"Reference": reference_runner},
        )

    with pytest.raises(ValueError, match="also fixed"):
        GridBenchmark(
            design_factory=TinyDesign,
            grid={"n": [25]},
            base_design_parameters={"n": 100},
            estimators={"Reference": reference_runner},
        )

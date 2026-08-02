import csv
from pathlib import Path

from orthogmm.reporting import (
    baseline_table,
    complexity_table,
    load_summary_rows,
    write_latex_table,
)


def write_fixture(path: Path) -> None:
    fieldnames = [
        "estimator",
        "parameter_index",
        "true_value",
        "repetitions",
        "successes",
        "success_rate",
        "bias",
        "rmse",
        "empirical_sd",
        "mean_standard_error",
        "coverage",
        "mean_runtime_seconds",
        "mean_objective_evaluations",
        "mean_demanding_evaluations",
        "n",
        "basis_k",
        "q_nodes",
        "failed_runs",
    ]
    rows = []
    for estimator, time, demand in (
        ("Initial", 0.1, 0.0),
        ("Full", 1.0, 45.0),
        ("SOP", 0.2, 6.0),
    ):
        for parameter in (0, 1):
            rows.append(
                {
                    "estimator": estimator,
                    "parameter_index": parameter,
                    "true_value": float(parameter),
                    "repetitions": 10,
                    "successes": 10,
                    "success_rate": 1.0,
                    "bias": 0.01,
                    "rmse": 0.02,
                    "empirical_sd": 0.02,
                    "mean_standard_error": 0.02,
                    "coverage": 0.95,
                    "mean_runtime_seconds": time,
                    "mean_objective_evaluations": 15.0,
                    "mean_demanding_evaluations": demand,
                    "n": 500,
                    "basis_k": 1,
                    "q_nodes": 40,
                    "failed_runs": 0,
                }
            )

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_baseline_table_and_latex(tmp_path: Path) -> None:
    source = tmp_path / "summary.csv"
    write_fixture(source)

    rows = load_summary_rows(source)
    table = baseline_table(rows)

    assert len(table) == 3
    assert table[1]["estimator"] == "Full"
    assert table[1]["demanding"] == 45.0

    latex = tmp_path / "table.tex"
    write_latex_table(
        table,
        latex,
        columns=["n", "estimator", "runtime"],
        headers=["n", "Estimator", "Time"],
    )

    text = latex.read_text(encoding="utf-8")
    assert "\\toprule" in text
    assert "Full" in text


def test_complexity_table(tmp_path: Path) -> None:
    source = tmp_path / "summary.csv"
    write_fixture(source)

    rows = load_summary_rows(source)
    table = complexity_table(
        rows,
        dimension="basis_k",
    )

    assert len(table) == 3
    assert table[2]["estimator"] == "SOP"
    assert table[2]["demanding_evaluations"] == 6.0

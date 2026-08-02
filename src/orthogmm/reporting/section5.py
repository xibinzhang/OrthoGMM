from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable


def load_summary_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load a Section 5 summary CSV with numeric type conversion."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)

    integer_fields = {
        "parameter_index",
        "repetitions",
        "successes",
        "n",
        "basis_k",
        "q_nodes",
        "failed_runs",
    }
    float_fields = {
        "true_value",
        "success_rate",
        "bias",
        "rmse",
        "empirical_sd",
        "mean_standard_error",
        "coverage",
        "mean_runtime_seconds",
        "mean_objective_evaluations",
        "mean_demanding_evaluations",
    }

    rows: list[dict[str, Any]] = []

    with source.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        required = {
            "estimator",
            "parameter_index",
            "bias",
            "rmse",
            "empirical_sd",
            "mean_standard_error",
            "coverage",
            "mean_runtime_seconds",
            "mean_demanding_evaluations",
            "success_rate",
            "n",
            "basis_k",
            "q_nodes",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Summary CSV is missing required columns: "
                + ", ".join(sorted(missing))
            )

        for raw in reader:
            row: dict[str, Any] = dict(raw)

            for name in integer_fields:
                value = row.get(name)
                if value not in (None, ""):
                    row[name] = int(float(value))

            for name in float_fields:
                value = row.get(name)
                if value not in (None, ""):
                    row[name] = float(value)

            rows.append(row)

    return rows


def _parameter_rows(
    rows: Iterable[dict[str, Any]],
) -> dict[tuple[int, str, int, int, int], dict[str, Any]]:
    indexed: dict[
        tuple[int, str, int, int, int],
        dict[str, Any],
    ] = {}

    for row in rows:
        key = (
            int(row["n"]),
            str(row["estimator"]),
            int(row["basis_k"]),
            int(row["q_nodes"]),
            int(row["parameter_index"]),
        )
        indexed[key] = row

    return indexed


def baseline_table(
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create the publication table for the baseline experiment."""

    rows = list(rows)
    indexed = _parameter_rows(rows)
    cells = sorted(
        {
            (
                int(row["n"]),
                str(row["estimator"]),
                int(row["basis_k"]),
                int(row["q_nodes"]),
            )
            for row in rows
        },
        key=lambda value: (
            value[0],
            ("Initial", "Full", "SOP").index(value[1]),
        ),
    )

    table: list[dict[str, Any]] = []

    for n, estimator, basis_k, q_nodes in cells:
        alpha = indexed[
            (n, estimator, basis_k, q_nodes, 0)
        ]
        beta = indexed[
            (n, estimator, basis_k, q_nodes, 1)
        ]

        table.append(
            {
                "n": n,
                "estimator": estimator,
                "bias_alpha": alpha["bias"],
                "rmse_alpha": alpha["rmse"],
                "sd_alpha": alpha["empirical_sd"],
                "se_alpha": alpha["mean_standard_error"],
                "coverage_alpha": alpha["coverage"],
                "bias_beta": beta["bias"],
                "rmse_beta": beta["rmse"],
                "sd_beta": beta["empirical_sd"],
                "se_beta": beta["mean_standard_error"],
                "coverage_beta": beta["coverage"],
                "runtime": alpha["mean_runtime_seconds"],
                "demanding": alpha[
                    "mean_demanding_evaluations"
                ],
                "success_rate": alpha["success_rate"],
            }
        )

    return table


def complexity_table(
    rows: Iterable[dict[str, Any]],
    *,
    dimension: str,
) -> list[dict[str, Any]]:
    """Create a compact computational-complexity table.

    Parameters
    ----------
    rows
        Summary rows.
    dimension
        Either ``"basis_k"`` or ``"q_nodes"``.
    """

    if dimension not in ("basis_k", "q_nodes"):
        raise ValueError(
            "dimension must be 'basis_k' or 'q_nodes'."
        )

    rows = list(rows)
    table: list[dict[str, Any]] = []

    cells = sorted(
        {
            (
                int(row[dimension]),
                str(row["estimator"]),
            )
            for row in rows
        },
        key=lambda value: (
            value[0],
            ("Initial", "Full", "SOP").index(value[1]),
        ),
    )

    for value, estimator in cells:
        matches = [
            row
            for row in rows
            if (
                int(row[dimension]) == value
                and row["estimator"] == estimator
                and int(row["parameter_index"]) == 0
            )
        ]

        if len(matches) != 1:
            raise ValueError(
                "Expected one parameter-zero summary row for "
                f"{dimension}={value}, estimator={estimator}."
            )

        row = matches[0]
        table.append(
            {
                dimension: value,
                "estimator": estimator,
                "runtime": row["mean_runtime_seconds"],
                "objective_evaluations": row[
                    "mean_objective_evaluations"
                ],
                "demanding_evaluations": row[
                    "mean_demanding_evaluations"
                ],
                "success_rate": row["success_rate"],
            }
        )

    return table


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def _format_value(
    value: Any,
    *,
    digits: int,
) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return _latex_escape(value)


def write_latex_table(
    rows: list[dict[str, Any]],
    path: str | Path,
    *,
    columns: list[str],
    headers: list[str] | None = None,
    digits: int = 4,
    caption: str | None = None,
    label: str | None = None,
) -> Path:
    """Write a booktabs-style LaTeX table."""

    if not rows:
        raise ValueError("rows must not be empty.")
    if headers is None:
        headers = columns
    if len(headers) != len(columns):
        raise ValueError(
            "headers and columns must have equal length."
        )

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    alignment = "l" + "r" * (len(columns) - 1)
    lines: list[str] = []

    if caption is not None or label is not None:
        lines.append(r"\begin{table}[!htbp]")
        lines.append(r"\centering")
        if caption is not None:
            lines.append(
                rf"\caption{{{_latex_escape(caption)}}}"
            )
        if label is not None:
            lines.append(
                rf"\label{{{_latex_escape(label)}}}"
            )

    lines.extend(
        [
            rf"\begin{{tabular}}{{{alignment}}}",
            r"\toprule",
            " & ".join(_latex_escape(item) for item in headers)
            + r" \\",
            r"\midrule",
        ]
    )

    for row in rows:
        lines.append(
            " & ".join(
                _format_value(row[column], digits=digits)
                for column in columns
            )
            + r" \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )

    if caption is not None or label is not None:
        lines.append(r"\end{table}")

    output.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return output


def _plot_metric(
    rows: list[dict[str, Any]],
    *,
    x_name: str,
    metric: str,
    ylabel: str,
    path: Path,
) -> Path:
    import matplotlib.pyplot as plt

    estimators = ("Initial", "Full", "SOP")

    figure = plt.figure()
    axis = figure.add_subplot(111)

    for estimator in estimators:
        subset = [
            row for row in rows
            if row["estimator"] == estimator
        ]
        subset.sort(key=lambda row: row[x_name])
        axis.plot(
            [row[x_name] for row in subset],
            [row[metric] for row in subset],
            marker="o",
            label=estimator,
        )

    axis.set_xlabel(x_name)
    axis.set_ylabel(ylabel)
    axis.legend()
    figure.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    return path


def generate_section5_outputs(
    *,
    baseline_csv: str | Path,
    basis_csv: str | Path,
    quadrature_csv: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """Generate Section 5 LaTeX tables and PDF figures."""

    output = Path(output_dir)
    baseline_rows = load_summary_rows(baseline_csv)
    basis_rows = load_summary_rows(basis_csv)
    quadrature_rows = load_summary_rows(quadrature_csv)

    baseline = baseline_table(baseline_rows)
    basis = complexity_table(
        basis_rows,
        dimension="basis_k",
    )
    quadrature = complexity_table(
        quadrature_rows,
        dimension="q_nodes",
    )

    created: list[Path] = []

    created.append(
        write_latex_table(
            baseline,
            output / "table_section5_baseline.tex",
            columns=[
                "n",
                "estimator",
                "bias_alpha",
                "rmse_alpha",
                "coverage_alpha",
                "bias_beta",
                "rmse_beta",
                "coverage_beta",
                "runtime",
                "demanding",
                "success_rate",
            ],
            headers=[
                "$n$",
                "Estimator",
                "Bias $\\alpha$",
                "RMSE $\\alpha$",
                "Cov. $\\alpha$",
                "Bias $\\beta$",
                "RMSE $\\beta$",
                "Cov. $\\beta$",
                "Time",
                "Demand",
                "Success",
            ],
            caption="Baseline Monte Carlo results.",
            label="tab:section5-baseline",
        )
    )

    created.append(
        write_latex_table(
            basis,
            output / "table_section5_basis.tex",
            columns=[
                "basis_k",
                "estimator",
                "runtime",
                "objective_evaluations",
                "demanding_evaluations",
                "success_rate",
            ],
            headers=[
                "$K$",
                "Estimator",
                "Time",
                "Objectives",
                "Demand",
                "Success",
            ],
            caption="Basis-complexity experiment.",
            label="tab:section5-basis",
        )
    )

    created.append(
        write_latex_table(
            quadrature,
            output / "table_section5_quadrature.tex",
            columns=[
                "q_nodes",
                "estimator",
                "runtime",
                "objective_evaluations",
                "demanding_evaluations",
                "success_rate",
            ],
            headers=[
                "$Q$",
                "Estimator",
                "Time",
                "Objectives",
                "Demand",
                "Success",
            ],
            caption="Quadrature-complexity experiment.",
            label="tab:section5-quadrature",
        )
    )

    baseline_complexity = [
        {
            "n": row["n"],
            "estimator": row["estimator"],
            "runtime": row["runtime"],
            "demanding_evaluations": row["demanding"],
        }
        for row in baseline
    ]

    created.append(
        _plot_metric(
            baseline_complexity,
            x_name="n",
            metric="runtime",
            ylabel="Mean runtime (seconds)",
            path=output / "figure_section5_runtime_n.pdf",
        )
    )
    created.append(
        _plot_metric(
            basis,
            x_name="basis_k",
            metric="runtime",
            ylabel="Mean runtime (seconds)",
            path=output / "figure_section5_runtime_k.pdf",
        )
    )
    created.append(
        _plot_metric(
            quadrature,
            x_name="q_nodes",
            metric="runtime",
            ylabel="Mean runtime (seconds)",
            path=output / "figure_section5_runtime_q.pdf",
        )
    )
    created.append(
        _plot_metric(
            basis,
            x_name="basis_k",
            metric="demanding_evaluations",
            ylabel="Mean demanding evaluations",
            path=output / "figure_section5_demand_k.pdf",
        )
    )
    created.append(
        _plot_metric(
            quadrature,
            x_name="q_nodes",
            metric="demanding_evaluations",
            ylabel="Mean demanding evaluations",
            path=output / "figure_section5_demand_q.pdf",
        )
    )

    return created

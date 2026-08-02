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
    estimator_order = ("Initial", "Full", "SOP")
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
            estimator_order.index(value[1]),
        ),
    )

    table: list[dict[str, Any]] = []

    for n, estimator, basis_k, q_nodes in cells:
        alpha = indexed[(n, estimator, basis_k, q_nodes, 0)]
        beta = indexed[(n, estimator, basis_k, q_nodes, 1)]

        table.append(
            {
                "n": n,
                "estimator": estimator,
                "bias_alpha": alpha["bias"],
                "rmse_alpha": alpha["rmse"],
                "coverage_alpha": alpha["coverage"],
                "bias_beta": beta["bias"],
                "rmse_beta": beta["rmse"],
                "coverage_beta": beta["coverage"],
                "runtime": alpha["mean_runtime_seconds"],
                "demanding": alpha["mean_demanding_evaluations"],
                "success_rate": alpha["success_rate"],
            }
        )

    return table


def complexity_comparison_table(
    rows: Iterable[dict[str, Any]],
    *,
    dimension: str,
) -> list[dict[str, Any]]:
    """Create one Full-versus-SOP row per complexity value."""

    if dimension not in ("basis_k", "q_nodes"):
        raise ValueError(
            "dimension must be 'basis_k' or 'q_nodes'."
        )

    rows = [
        row
        for row in rows
        if int(row["parameter_index"]) == 0
        and row["estimator"] in ("Full", "SOP")
    ]

    values = sorted({int(row[dimension]) for row in rows})
    table: list[dict[str, Any]] = []

    for value in values:
        matched = {
            row["estimator"]: row
            for row in rows
            if int(row[dimension]) == value
        }

        if set(matched) != {"Full", "SOP"}:
            raise ValueError(
                f"Expected Full and SOP rows for {dimension}={value}."
            )

        full = matched["Full"]
        sop = matched["SOP"]

        full_time = float(full["mean_runtime_seconds"])
        sop_time = float(sop["mean_runtime_seconds"])
        full_demand = float(full["mean_demanding_evaluations"])
        sop_demand = float(sop["mean_demanding_evaluations"])

        table.append(
            {
                dimension: value,
                "full_runtime": full_time,
                "sop_runtime": sop_time,
                "speedup": full_time / sop_time,
                "full_demanding": full_demand,
                "sop_demanding": sop_demand,
                "demand_reduction": 1.0 - sop_demand / full_demand,
                "full_success": float(full["success_rate"]),
                "sop_success": float(sop["success_rate"]),
            }
        )

    return table


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
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
    alignment: str | None = None,
    raw_headers: bool = False,
    font_command: str | None = None,
    tabcolsep: float | None = None,
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

    if alignment is None:
        alignment = "l" + "r" * (len(columns) - 1)

    if len(alignment) != len(columns):
        raise ValueError(
            "alignment must contain one specifier per column."
        )

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    if caption is not None or label is not None:
        lines.append(r"\begin{table}[!htbp]")
        lines.append(r"\centering")

        if caption is not None:
            lines.append(
                rf"\caption{{{_latex_escape(caption)}}}"
            )

        if label is not None:
            lines.append(rf"\label{{{label}}}")

    if font_command is not None:
        lines.append(font_command)

    if tabcolsep is not None:
        lines.append(
            rf"\setlength{{\tabcolsep}}{{{tabcolsep}pt}}"
        )

    header_values = (
        headers
        if raw_headers
        else [_latex_escape(item) for item in headers]
    )

    lines.extend(
        [
            rf"\begin{{tabular}}{{{alignment}}}",
            r"\toprule",
            " & ".join(header_values) + r" \\",
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
    x_label: str,
    metric: str,
    ylabel: str,
    path: Path,
) -> Path:
    import matplotlib.pyplot as plt

    figure = plt.figure()
    axis = figure.add_subplot(111)

    for estimator in ("Full", "SOP"):
        subset = [
            row
            for row in rows
            if row["estimator"] == estimator
        ]
        subset.sort(key=lambda row: row[x_name])
        axis.plot(
            [row[x_name] for row in subset],
            [row[metric] for row in subset],
            marker="o",
            label=estimator,
        )

    axis.set_xlabel(x_label)
    axis.set_ylabel(ylabel)
    axis.legend()
    figure.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    return path


def _complexity_plot_rows(
    rows: list[dict[str, Any]],
    *,
    dimension: str,
) -> list[dict[str, Any]]:
    return [
        {
            dimension: int(row[dimension]),
            "estimator": row["estimator"],
            "runtime": float(row["mean_runtime_seconds"]),
            "demanding": float(
                row["mean_demanding_evaluations"]
            ),
        }
        for row in rows
        if int(row["parameter_index"]) == 0
        and row["estimator"] in ("Full", "SOP")
    ]


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
    basis = complexity_comparison_table(
        basis_rows,
        dimension="basis_k",
    )
    quadrature = complexity_comparison_table(
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
            alignment="rlrrrrrrrrr",
            raw_headers=True,
            font_command=r"\scriptsize",
            tabcolsep=3.0,
        )
    )

    created.append(
        write_latex_table(
            basis,
            output / "table_section5_basis.tex",
            columns=[
                "basis_k",
                "full_runtime",
                "sop_runtime",
                "speedup",
                "full_demanding",
                "sop_demanding",
                "demand_reduction",
                "full_success",
                "sop_success",
            ],
            headers=[
                "$K$",
                "Full time",
                "SOP time",
                "Speedup",
                "Full demand",
                "SOP demand",
                "Reduction",
                "Full success",
                "SOP success",
            ],
            caption="Basis-complexity experiment.",
            label="tab:section5-basis",
            alignment="rrrrrrrrr",
            raw_headers=True,
            font_command=r"\small",
            tabcolsep=4.0,
        )
    )

    created.append(
        write_latex_table(
            quadrature,
            output / "table_section5_quadrature.tex",
            columns=[
                "q_nodes",
                "full_runtime",
                "sop_runtime",
                "speedup",
                "full_demanding",
                "sop_demanding",
                "demand_reduction",
                "full_success",
                "sop_success",
            ],
            headers=[
                "$Q$",
                "Full time",
                "SOP time",
                "Speedup",
                "Full demand",
                "SOP demand",
                "Reduction",
                "Full success",
                "SOP success",
            ],
            caption="Quadrature-complexity experiment.",
            label="tab:section5-quadrature",
            alignment="rrrrrrrrr",
            raw_headers=True,
            font_command=r"\small",
            tabcolsep=4.0,
        )
    )

    baseline_plot = [
        {
            "n": row["n"],
            "estimator": row["estimator"],
            "runtime": row["runtime"],
            "demanding": row["demanding"],
        }
        for row in baseline
        if row["estimator"] in ("Full", "SOP")
    ]

    basis_plot = _complexity_plot_rows(
        basis_rows,
        dimension="basis_k",
    )
    quadrature_plot = _complexity_plot_rows(
        quadrature_rows,
        dimension="q_nodes",
    )

    created.append(
        _plot_metric(
            baseline_plot,
            x_name="n",
            x_label="$n$",
            metric="runtime",
            ylabel="Mean runtime (seconds)",
            path=output / "figure_section5_runtime_n.pdf",
        )
    )
    created.append(
        _plot_metric(
            basis_plot,
            x_name="basis_k",
            x_label="$K$",
            metric="runtime",
            ylabel="Mean runtime (seconds)",
            path=output / "figure_section5_runtime_k.pdf",
        )
    )
    created.append(
        _plot_metric(
            quadrature_plot,
            x_name="q_nodes",
            x_label="$Q$",
            metric="runtime",
            ylabel="Mean runtime (seconds)",
            path=output / "figure_section5_runtime_q.pdf",
        )
    )
    created.append(
        _plot_metric(
            basis_plot,
            x_name="basis_k",
            x_label="$K$",
            metric="demanding",
            ylabel="Mean demanding evaluations",
            path=output / "figure_section5_demand_k.pdf",
        )
    )
    created.append(
        _plot_metric(
            quadrature_plot,
            x_name="q_nodes",
            x_label="$Q$",
            metric="demanding",
            ylabel="Mean demanding evaluations",
            path=output / "figure_section5_demand_q.pdf",
        )
    )

    return created

"""Generic reporting utilities for Monte Carlo benchmark results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

if TYPE_CHECKING:
    from orthogmm.simulation.experiment import ParameterSummary


_DEFAULT_COLUMNS = (
    "estimator",
    "parameter_index",
    "true_value",
    "bias",
    "rmse",
    "empirical_sd",
    "mean_standard_error",
    "coverage",
    "success_rate",
    "mean_runtime_seconds",
    "mean_demanding_evaluations",
)

_DEFAULT_HEADERS = {
    "estimator": "Estimator",
    "parameter_index": "Parameter",
    "true_value": "True",
    "bias": "Bias",
    "rmse": "RMSE",
    "empirical_sd": "SD",
    "mean_standard_error": "Mean SE",
    "coverage": "Coverage",
    "success_rate": "Success",
    "mean_runtime_seconds": "Time (s)",
    "mean_objective_evaluations": "Objective evals",
    "mean_demanding_evaluations": "Demanding evals",
    "repetitions": "Replications",
    "successes": "Successful",
}


def summary_rows(
    summaries: Iterable[ParameterSummary],
) -> list[dict[str, Any]]:
    """Convert parameter summaries to plain dictionaries."""

    return [
        {
            "estimator": row.estimator,
            "parameter_index": row.parameter_index,
            "true_value": row.true_value,
            "repetitions": row.repetitions,
            "successes": row.successes,
            "success_rate": row.success_rate,
            "bias": row.bias,
            "rmse": row.rmse,
            "empirical_sd": row.empirical_sd,
            "mean_standard_error": row.mean_standard_error,
            "coverage": row.coverage,
            "mean_runtime_seconds": row.mean_runtime_seconds,
            "mean_objective_evaluations": row.mean_objective_evaluations,
            "mean_demanding_evaluations": row.mean_demanding_evaluations,
        }
        for row in summaries
    ]


def format_summary_table(
    rows: Sequence[dict[str, Any]],
    *,
    columns: Sequence[str] = _DEFAULT_COLUMNS,
    digits: int = 4,
) -> str:
    """Format summary rows as a compact plain-text table."""

    if not rows:
        raise ValueError("Cannot format an empty summary table.")
    if digits < 0:
        raise ValueError("digits must be non-negative.")

    headers = [_DEFAULT_HEADERS.get(column, column) for column in columns]
    values = [
        [_format_value(row.get(column), column=column, digits=digits) for column in columns]
        for row in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in values))
        for index in range(len(columns))
    ]

    def render(items: Sequence[str]) -> str:
        cells = []
        for index, item in enumerate(items):
            if columns[index] == "estimator":
                cells.append(item.ljust(widths[index]))
            else:
                cells.append(item.rjust(widths[index]))
        return "  ".join(cells)

    separator = "  ".join("-" * width for width in widths)
    return "\n".join([render(headers), separator, *(render(row) for row in values)])


def write_summary_latex(
    rows: Sequence[dict[str, Any]],
    path: str | Path,
    *,
    columns: Sequence[str] = _DEFAULT_COLUMNS,
    digits: int = 4,
    caption: str | None = None,
    label: str | None = None,
) -> Path:
    """Write a booktabs-compatible LaTeX table."""

    if not rows:
        raise ValueError("Cannot export an empty summary table.")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    alignment = "l" + "r" * (len(columns) - 1)
    headers = [_DEFAULT_HEADERS.get(column, column) for column in columns]

    lines = ["\\begin{table}[!htbp]", "\\centering"]
    if caption:
        lines.append(f"\\caption{{{_latex_escape(caption)}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.extend(
        [
            f"\\begin{{tabular}}{{{alignment}}}",
            "\\toprule",
            " & ".join(_latex_escape(header) for header in headers) + r" \\",
            "\\midrule",
        ]
    )

    for row in rows:
        cells = [
            _latex_escape(_format_value(row.get(column), column=column, digits=digits))
            for column in columns
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def plot_metric(
    rows: Sequence[dict[str, Any]],
    path: str | Path,
    *,
    metric: str,
    ylabel: str,
) -> Path:
    """Plot an estimator-level metric averaged over parameters."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Plotting requires matplotlib. Install OrthoGMM with "
            "the 'reporting' extra or install matplotlib directly."
        ) from exc

    aggregates: dict[str, list[float]] = {}
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        aggregates.setdefault(str(row["estimator"]), []).append(float(value))

    if not aggregates:
        raise ValueError(f"No finite values are available for {metric!r}.")

    names = list(aggregates)
    values = [sum(aggregates[name]) / len(aggregates[name]) for name in names]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots()
    axis.bar(names, values)
    axis.set_ylabel(ylabel)
    axis.set_xlabel("Estimator")
    figure.tight_layout()
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)
    return destination


def _format_value(value: Any, *, column: str, digits: int) -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if column in {"coverage", "success_rate"}:
            return f"{value:.{digits}f}"
        return f"{value:.{digits}f}"
    return str(value)


def _latex_escape(value: str) -> str:
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
    return "".join(replacements.get(character, character) for character in value)

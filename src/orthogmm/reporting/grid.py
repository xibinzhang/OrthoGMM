"""Reporting utilities for multi-design Monte Carlo grid benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


_HEADERS = {
    "cell_index": "Cell",
    "cell_seed": "Cell seed",
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
    "candidate": "Candidate",
    "reference": "Reference",
    "pairs": "Pairs",
    "joint_success_rate": "Joint success",
    "mean_parameter_distance": "Mean parameter distance",
    "p95_parameter_distance": "P95 parameter distance",
    "mean_covariance_distance": "Mean covariance distance",
    "mean_runtime_speedup": "Runtime speedup",
    "mean_demanding_evaluation_reduction": "Demanding-eval reduction",
}

_PERCENT_COLUMNS = {
    "joint_success_rate",
    "convergence_agreement_rate",
    "mean_demanding_evaluation_reduction",
}


def format_grid_table(
    rows: Sequence[dict[str, Any]],
    *,
    columns: Sequence[str],
    digits: int = 4,
) -> str:
    """Format flattened grid rows as a compact plain-text table."""

    if not rows:
        raise ValueError("Cannot format an empty grid table.")
    if not columns:
        raise ValueError("columns cannot be empty.")
    if digits < 0:
        raise ValueError("digits must be non-negative.")

    headers = [_HEADERS.get(column, column) for column in columns]
    values = [
        [
            _format_value(row.get(column), column=column, digits=digits)
            for column in columns
        ]
        for row in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in values))
        for index in range(len(columns))
    ]

    def render(items: Sequence[str]) -> str:
        cells = []
        for index, item in enumerate(items):
            column = columns[index]
            if column in {"estimator", "candidate", "reference"}:
                cells.append(item.ljust(widths[index]))
            else:
                cells.append(item.rjust(widths[index]))
        return "  ".join(cells)

    separator = "  ".join("-" * width for width in widths)
    return "\n".join(
        [render(headers), separator, *(render(row) for row in values)]
    )


def write_grid_latex(
    rows: Sequence[dict[str, Any]],
    path: str | Path,
    *,
    columns: Sequence[str],
    digits: int = 4,
    caption: str | None = None,
    label: str | None = None,
) -> Path:
    """Write flattened grid rows as a booktabs-compatible LaTeX table."""

    if not rows:
        raise ValueError("Cannot export an empty grid table.")
    if not columns:
        raise ValueError("columns cannot be empty.")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    alignment = "l" + "r" * (len(columns) - 1)
    headers = [_HEADERS.get(column, column) for column in columns]

    lines = ["\\begin{table}[!htbp]", "\\centering"]
    if caption:
        lines.append(f"\\caption{{{_latex_escape(caption)}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.extend(
        [
            f"\\begin{{tabular}}{{{alignment}}}",
            "\\toprule",
            " & ".join(_latex_escape(header) for header in headers)
            + r" \\",
            "\\midrule",
        ]
    )

    for row in rows:
        cells = [
            _latex_escape(
                _format_value(row.get(column), column=column, digits=digits)
            )
            for column in columns
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def plot_grid_metric(
    rows: Sequence[dict[str, Any]],
    path: str | Path,
    *,
    x: str,
    metric: str,
    series_columns: Sequence[str],
    ylabel: str | None = None,
) -> Path:
    """Plot a metric over one grid dimension, grouped into named series."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Plotting requires matplotlib. Install OrthoGMM with "
            "the 'reporting' extra or install matplotlib directly."
        ) from exc

    usable = [
        row
        for row in rows
        if row.get(x) is not None and _is_finite_number(row.get(metric))
    ]
    if not usable:
        raise ValueError(
            f"No finite values are available for metric {metric!r} "
            f"against grid parameter {x!r}."
        )

    grouped: dict[tuple[Any, ...], list[tuple[Any, float]]] = {}
    for row in usable:
        key = tuple(row.get(column) for column in series_columns)
        grouped.setdefault(key, []).append((row[x], float(row[metric])))

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots()

    for key, points in grouped.items():
        points = sorted(points, key=lambda item: _sort_key(item[0]))
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        label = _series_label(series_columns, key)
        axis.plot(x_values, y_values, marker="o", label=label or None)

    axis.set_xlabel(_HEADERS.get(x, x))
    axis.set_ylabel(ylabel or _HEADERS.get(metric, metric))
    if any(_series_label(series_columns, key) for key in grouped):
        axis.legend()
    figure.tight_layout()
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)
    return destination


def _series_label(columns: Sequence[str], key: tuple[Any, ...]) -> str:
    return ", ".join(
        f"{_HEADERS.get(column, column)}={value}"
        for column, value in zip(columns, key)
    )


def _format_value(value: Any, *, column: str, digits: int) -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if column in _PERCENT_COLUMNS:
            return f"{100.0 * value:.{digits}f}%"
        return f"{value:.{digits}f}"
    return str(value)


def _is_finite_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and abs(number) != float("inf")


def _sort_key(value: Any) -> tuple[int, Any]:
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


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

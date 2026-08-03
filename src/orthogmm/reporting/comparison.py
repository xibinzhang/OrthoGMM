"""Reporting utilities for paired estimator comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

if TYPE_CHECKING:
    from orthogmm.simulation.comparison import ComparisonRecord, ComparisonSummary


_DEFAULT_COMPARISON_COLUMNS = (
    "candidate",
    "reference",
    "pairs",
    "joint_success_rate",
    "mean_parameter_distance",
    "p95_parameter_distance",
    "mean_covariance_distance",
    "mean_absolute_objective_difference",
    "mean_runtime_speedup",
    "mean_demanding_evaluation_reduction",
)

_HEADERS = {
    "candidate": "Candidate",
    "reference": "Reference",
    "pairs": "Pairs",
    "joint_success_rate": "Joint success",
    "convergence_agreement_rate": "Convergence agreement",
    "mean_parameter_distance": "Mean parameter distance",
    "median_parameter_distance": "Median parameter distance",
    "p95_parameter_distance": "P95 parameter distance",
    "max_parameter_distance": "Max parameter distance",
    "mean_covariance_distance": "Mean covariance distance",
    "mean_objective_difference": "Mean objective difference",
    "mean_absolute_objective_difference": "Mean absolute objective gap",
    "mean_runtime_speedup": "Runtime speedup",
    "mean_demanding_evaluation_reduction": "Demanding-eval reduction",
}


def format_comparison_table(
    summary: ComparisonSummary,
    *,
    columns: Sequence[str] = _DEFAULT_COMPARISON_COLUMNS,
    digits: int = 4,
) -> str:
    """Format one comparison summary as a compact text table."""

    if digits < 0:
        raise ValueError("digits must be non-negative.")

    row = summary.__dict__
    headers = [_HEADERS.get(column, column) for column in columns]
    values = [_format_value(row.get(column), column=column, digits=digits) for column in columns]
    widths = [max(len(header), len(value)) for header, value in zip(headers, values)]

    header_line = "  ".join(
        header.ljust(width) if index < 2 else header.rjust(width)
        for index, (header, width) in enumerate(zip(headers, widths))
    )
    separator = "  ".join("-" * width for width in widths)
    value_line = "  ".join(
        value.ljust(width) if index < 2 else value.rjust(width)
        for index, (value, width) in enumerate(zip(values, widths))
    )
    return "\n".join((header_line, separator, value_line))


def write_comparison_latex(
    summary: ComparisonSummary,
    path: str | Path,
    *,
    columns: Sequence[str] = _DEFAULT_COMPARISON_COLUMNS,
    digits: int = 4,
    caption: str | None = None,
    label: str | None = None,
) -> Path:
    """Write a booktabs-compatible comparison table."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    row = summary.__dict__
    alignment = "ll" + "r" * max(0, len(columns) - 2)

    lines = ["\\begin{table}[!htbp]", "\\centering"]
    if caption:
        lines.append(f"\\caption{{{_latex_escape(caption)}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.extend(
        [
            f"\\begin{{tabular}}{{{alignment}}}",
            "\\toprule",
            " & ".join(_latex_escape(_HEADERS.get(column, column)) for column in columns)
            + r" \\",
            "\\midrule",
            " & ".join(
                _latex_escape(_format_value(row.get(column), column=column, digits=digits))
                for column in columns
            )
            + r" \\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def plot_comparison_metric(
    records: Sequence[ComparisonRecord],
    path: str | Path,
    *,
    metric: str,
    xlabel: str,
) -> Path:
    """Plot a histogram of one finite paired-comparison metric."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Plotting requires matplotlib. Install OrthoGMM with "
            "the 'reporting' extra or install matplotlib directly."
        ) from exc

    values = np.asarray(
        [
            float(value)
            for record in records
            if (value := getattr(record, metric)) is not None and np.isfinite(value)
        ],
        dtype=float,
    )
    if not values.size:
        raise ValueError(f"No finite values are available for {metric!r}.")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots()
    axis.hist(values, bins="auto")
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Frequency")
    figure.tight_layout()
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)
    return destination


def _format_value(value: Any, *, column: str, digits: int) -> str:
    if value is None:
        return "--"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if column in {
            "joint_success_rate",
            "convergence_agreement_rate",
            "mean_demanding_evaluation_reduction",
        }:
            return f"{100.0 * value:.{digits}f}%"
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

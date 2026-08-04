"""Reporting utilities for reproducible OrthoGMM experiments."""

from .comparison import (
    format_comparison_table,
    plot_comparison_metric,
    write_comparison_latex,
)
from .benchmark import (
    format_summary_table,
    plot_metric,
    summary_rows,
    write_summary_latex,
)
from .grid import (
    format_grid_table,
    plot_grid_metric,
    write_grid_latex,
)
from .section5 import (
    baseline_table,
    complexity_comparison_table,
    generate_section5_outputs,
    load_summary_rows,
    write_latex_table,
)

complexity_table = complexity_comparison_table

__all__ = [
    "baseline_table",
    "complexity_comparison_table",
    "complexity_table",
    "format_comparison_table",
    "format_summary_table",
    "format_grid_table",
    "generate_section5_outputs",
    "load_summary_rows",
    "plot_comparison_metric",
    "plot_metric",
    "plot_grid_metric",
    "summary_rows",
    "write_comparison_latex",
    "write_latex_table",
    "write_summary_latex",
    "write_grid_latex",
]

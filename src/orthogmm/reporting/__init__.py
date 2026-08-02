"""Reporting utilities for reproducible OrthoGMM experiments."""

from .section5 import (
    baseline_table,
    complexity_comparison_table,
    generate_section5_outputs,
    load_summary_rows,
    write_latex_table,
)

# Backward-compatible alias for the original compact representation.
complexity_table = complexity_comparison_table

__all__ = [
    "baseline_table",
    "complexity_comparison_table",
    "complexity_table",
    "generate_section5_outputs",
    "load_summary_rows",
    "write_latex_table",
]

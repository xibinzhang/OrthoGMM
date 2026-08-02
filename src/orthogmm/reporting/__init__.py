"""Reporting utilities for reproducible OrthoGMM experiments."""

from .section5 import (
    baseline_table,
    complexity_table,
    generate_section5_outputs,
    load_summary_rows,
    write_latex_table,
)

__all__ = [
    "baseline_table",
    "complexity_table",
    "generate_section5_outputs",
    "load_summary_rows",
    "write_latex_table",
]

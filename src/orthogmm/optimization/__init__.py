"""Optimization utilities for OrthoGMM."""

from .trust_region import (
    QuadraticTrustRegion,
    TrustRegionResult,
)

__all__ = [
    "QuadraticTrustRegion",
    "TrustRegionResult",
]

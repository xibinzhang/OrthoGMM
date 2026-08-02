"""Smoke test for the multi-fidelity BLP adapter.

This example uses a small synthetic subclass. It verifies that the generic
OrthoGMM engine can estimate a model whose tractable moments come from a
low-fidelity BLP computation and whose demanding moments are the high-minus-low
correction.

Run from the repository root with:

    py examples\blp_smoke.py
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from orthogmm import (
    FidelityConfig,
    MultiFidelityBLPModel,
    fit_full_gmm,
    fit_seip,
    fit_tractable_gmm,
)


class SyntheticBLPModel(MultiFidelityBLPModel):
    def __init__(self, seed: int = 123, n_markets: int = 1000):
        rng = np.random.default_rng(seed)

        self.low_fidelity = FidelityConfig(
            name="low",
            draws=10,
            contraction_tolerance=1e-6,
            max_iterations=100,
            seed=7,
        )
        self.high_fidelity = FidelityConfig(
            name="high",
            draws=100,
            contraction_tolerance=1e-12,
            max_iterations=1000,
            seed=7,
        )

        self.x = rng.normal(size=(n_markets, 2))
        self.zg = rng.normal(size=(n_markets, 2))
        self.zh = rng.normal(size=(n_markets, 2))
        self.truth = np.array([1.0, -0.5])
        self.y = self.x @ self.truth + rng.normal(
            scale=0.25,
            size=n_markets,
        )

    def moments_at_fidelity(self, theta, fidelity):
        residual = self.y - self.x @ np.asarray(theta)
        low = self.zg * residual[:, None]

        if fidelity.name == "low":
            return low

        if fidelity.name == "high":
            return low + self.zh * residual[:, None]

        raise ValueError(f"Unknown fidelity: {fidelity.name}")


def timed(function, *args, **kwargs):
    start = perf_counter()
    result = function(*args, **kwargs)
    return result, perf_counter() - start


def main() -> None:
    model = SyntheticBLPModel()
    theta0 = np.zeros(2)

    initial, initial_time = timed(
        fit_tractable_gmm,
        model,
        theta0,
    )
    full, full_time = timed(
        fit_full_gmm,
        model,
        theta0,
    )
    sop, sop_time = timed(
        fit_seip,
        model,
        theta0,
    )

    print("Synthetic multi-fidelity BLP smoke test")
    print(f"Truth: {model.truth}")
    print()

    for name, result, runtime in (
        ("Initial", initial, initial_time),
        ("Full", full, full_time),
        ("SOP", sop, sop_time),
    ):
        print(
            f"{name:<8} "
            f"theta={np.array2string(result.theta, precision=6)} "
            f"time={runtime:.4f}s "
            f"demanding={result.counts.demanding_moments_total} "
            f"success={result.success}"
        )

    print()
    print(
        "Distance between Full and SOP: "
        f"{np.linalg.norm(full.theta - sop.theta):.6g}"
    )


if __name__ == "__main__":
    main()

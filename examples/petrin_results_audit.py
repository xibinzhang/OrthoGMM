"""Audit public PyBLP objects needed for market-level moment extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from orthogmm import FidelityConfig
from orthogmm.model.petrin import build_petrin_problem
from orthogmm.solvers import PyBLPSolver


def describe_attribute(obj, name: str) -> None:
    if not hasattr(obj, name):
        print(f"{name:<32} MISSING")
        return

    value = getattr(obj, name)

    if value is None:
        print(f"{name:<32} None")
        return

    try:
        array = np.asarray(value)
        shape = str(array.shape)
        dtype = str(array.dtype)

        finite = "n/a"
        if np.issubdtype(array.dtype, np.number):
            finite = str(bool(np.all(np.isfinite(array.astype(float)))))

        print(
            f"{name:<32}"
            f"shape={shape:<18}"
            f"dtype={dtype:<12}"
            f"finite={finite}"
        )
    except Exception as error:
        print(
            f"{name:<32}"
            f"type={type(value).__name__:<20}"
            f"array conversion failed: {error}"
        )


def main() -> None:
    setup = build_petrin_problem()

    fidelity = FidelityConfig(
        name="aggregate_fixed",
        draws=13000,
        contraction_tolerance=1e-10,
        max_iterations=1000,
        seed=0,
    )

    evaluation = PyBLPSolver().solve(
        setup,
        fidelity=fidelity,
        include_micro=False,
        fixed_parameters=True,
        method="1s",
    )

    results = evaluation.results

    print("\n" + "=" * 72)
    print("Object classes")
    print("=" * 72)
    print(f"Problem type: {type(setup.problem)}")
    print(f"Results type: {type(results)}")
    print("Results MRO:")
    for cls in results.__class__.__mro__:
        print(f"  {cls}")

    print("\n" + "=" * 72)
    print("Problem dimensions")
    print("=" * 72)
    print(f"Markets:  {setup.n_markets}")
    print(f"Products: {setup.n_products}")
    print(f"Agents:   {setup.n_agents}")

    print("\n" + "=" * 72)
    print("Selected public ProblemResults attributes")
    print("=" * 72)

    attributes = (
        "theta",
        "moments",
        "moments_jacobian",
        "moments_covariances",
        "W",
        "xi",
        "omega",
        "beta",
        "gamma",
        "sigma",
        "pi",
        "micro",
        "micro_values",
        "objective",
        "gradient",
        "delta",
        "shares",
        "costs",
    )

    for name in attributes:
        describe_attribute(results, name)

    print("\n" + "=" * 72)
    print("Market structure")
    print("=" * 72)

    product_market_ids = np.asarray(
        setup.product_data["market_ids"]
    )
    unique_markets, counts = np.unique(
        product_market_ids,
        return_counts=True,
    )

    print(f"market_ids shape: {product_market_ids.shape}")
    print(f"Unique markets:   {unique_markets.size}")
    print("\nProducts per market:")

    for market_id, count in zip(unique_markets, counts):
        print(f"  {market_id}: {count}")

    print("\n" + "=" * 72)
    print("Public result attributes containing useful keywords")
    print("=" * 72)

    keywords = (
        "moment",
        "micro",
        "market",
        "product",
        "agent",
        "xi",
        "omega",
        "residual",
        "covariance",
        "jacobian",
        "share",
        "delta",
    )

    names = sorted(
        name
        for name in dir(results)
        if not name.startswith("_")
        and any(
            keyword in name.lower()
            for keyword in keywords
        )
    )

    for name in names:
        print(name)

    print("\n" + "=" * 72)
    print("Problem public attributes containing useful keywords")
    print("=" * 72)

    problem_names = sorted(
        name
        for name in dir(setup.problem)
        if not name.startswith("_")
        and any(
            keyword in name.lower()
            for keyword in keywords
        )
    )

    for name in problem_names:
        print(name)


if __name__ == "__main__":
    main()

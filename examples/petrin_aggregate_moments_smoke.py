"""Verify real Petrin market-level aggregate IV contributions."""

from __future__ import annotations

import numpy as np

from orthogmm import FidelityConfig
from orthogmm.model.petrin import build_petrin_problem
from orthogmm.moments import AggregateIVMomentBuilder
from orthogmm.solvers import PyBLPSolver


def main() -> None:
    setup = build_petrin_problem()

    fidelity = FidelityConfig(
        name="aggregate_fixed",
        draws=13000,
        contraction_tolerance=1e-10,
        max_iterations=1000,
    )

    evaluation = PyBLPSolver().solve(
        setup,
        fidelity=fidelity,
        include_micro=False,
        fixed_parameters=True,
        method="1s",
    )
    results = evaluation.results

    moments = AggregateIVMomentBuilder().from_pyblp(
        setup.problem,
        results,
    )

    public_average = np.asarray(
        results.moments,
        dtype=float,
    ).reshape(-1)

    discrepancy = moments.average - public_average

    print("Petrin aggregate IV market moments")
    print(f"Market matrix shape: {moments.combined.shape}")
    print(f"Demand block shape:  {moments.demand.shape}")
    print(f"Supply block shape:  {moments.supply.shape}")
    print(f"Public moments:      {public_average.shape}")
    print(
        "Maximum absolute aggregation discrepancy: "
        f"{np.max(np.abs(discrepancy)):.6e}"
    )

    np.testing.assert_allclose(
        moments.average,
        public_average,
        rtol=1e-8,
        atol=1e-10,
    )


if __name__ == "__main__":
    main()

"""Run one fixed-parameter aggregate PyBLP evaluation.

This is intentionally not an OrthoGMM estimation. It validates the solver
boundary before market-level moment extraction is implemented.
"""

from orthogmm import FidelityConfig
from orthogmm.model.petrin import build_petrin_problem
from orthogmm.solvers import PyBLPSolver


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

    print("Petrin fixed-parameter aggregate evaluation")
    print(f"Elapsed seconds: {evaluation.elapsed_seconds:.3f}")
    print(f"Theta dimension: {len(results.theta)}")
    print(f"Moment count: {len(results.moments)}")
    print(f"Objective: {float(results.objective):.6g}")
    print(
        "Contraction evaluations: "
        f"{int(results.cumulative_contraction_evaluations.sum())}"
    )


if __name__ == "__main__":
    main()

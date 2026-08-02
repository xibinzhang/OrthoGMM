"""Run the first real Petrin application-model evaluation."""

from __future__ import annotations

import numpy as np

from orthogmm.applications.petrin import PetrinApplicationModel


def main() -> None:
    model = PetrinApplicationModel()

    print("Petrin application model")
    print(f"Active parameter dimension: {model.parameter_dimension}")
    print(f"Initial theta: {model.theta0}")

    evaluation = model.evaluate_aggregate(model.theta0)
    moments = evaluation.moments

    public = np.asarray(
        evaluation.pyblp.results.moments,
        dtype=float,
    ).reshape(-1)

    print(f"Market moments shape: {moments.combined.shape}")
    print(
        "Maximum aggregation discrepancy: "
        f"{np.max(np.abs(moments.average - public)):.6e}"
    )
    print(
        "Fixed evaluation time: "
        f"{evaluation.pyblp.elapsed_seconds:.3f}s"
    )


if __name__ == "__main__":
    main()

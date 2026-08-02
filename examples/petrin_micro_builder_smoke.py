"""Construct all 13 Petrin market-level micro contributions.

This performs 14 fixed PyBLP micro evaluations and may take several minutes.
"""

from __future__ import annotations

import numpy as np

from orthogmm.model.petrin import build_petrin_problem
from orthogmm.operators.blp_micro import (
    PetrinMicroJackknifeBuilder,
)


def main() -> None:
    setup = build_petrin_problem()

    result = PetrinMicroJackknifeBuilder().build(
        setup,
        sigma=setup.initial_sigma,
        pi=setup.initial_pi,
    )

    print("Petrin micro jackknife representation")
    print(f"Matrix shape: {result.contributions.shape}")
    print(
        "Maximum mean discrepancy: "
        f"{np.max(np.abs(result.contributions.mean(axis=0) - result.full_residual)):.6e}"
    )
    print(
        "Raw pseudo-value mean discrepancy: "
        f"{np.max(np.abs(result.raw_pseudo_values.mean(axis=0) - result.full_residual)):.6e}"
    )
    print(
        "Centering adjustment norm: "
        f"{np.linalg.norm(result.centering_adjustment):.6e}"
    )
    print(
        "Total elapsed seconds: "
        f"{result.total_elapsed_seconds:.3f}"
    )
    print("Full micro residual:")
    print(
        np.array2string(
            result.full_residual,
            precision=8,
            suppress_small=False,
        )
    )


if __name__ == "__main__":
    main()

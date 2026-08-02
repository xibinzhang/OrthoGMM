"""Build the complete local Petrin SOP state.

This performs the aggregate evaluation, thirteen leave-one-market-out micro
evaluations, and one additional full micro evaluation for the analytical
micro Jacobian. It may take several minutes.
"""

from __future__ import annotations

import numpy as np

from orthogmm.applications.petrin import (
    PetrinApplicationModel,
    PetrinLocalStateBuilder,
)


def main() -> None:
    model = PetrinApplicationModel()
    state = PetrinLocalStateBuilder(model).build(model.theta0)

    print("Petrin local SOP state")
    print(f"Markets:              {state.n_markets}")
    print(
        f"Tractable moments:    {state.tractable_moments.shape}"
    )
    print(
        f"Demanding moments:    {state.demanding_moments.shape}"
    )
    print(
        f"Tractable Jacobian:   {state.tractable_jacobian.shape}"
    )
    print(
        f"Demanding Jacobian:   {state.demanding_jacobian.shape}"
    )
    print(
        "Demanding mean norm: "
        f"{np.linalg.norm(state.demanding_moments.mean(axis=0)):.6e}"
    )
    print(
        "Total elapsed seconds: "
        f"{state.total_elapsed_seconds:.3f}"
    )


if __name__ == "__main__":
    main()

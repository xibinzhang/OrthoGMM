"""Build Petrin projected-information objects from the local SOP state.

This script rebuilds the local Petrin state and therefore may take several
minutes.
"""

from __future__ import annotations

import numpy as np

from orthogmm.applications.petrin import (
    PetrinApplicationModel,
    PetrinLocalStateBuilder,
)
from orthogmm.operators import ProjectedInformationOperator


def main() -> None:
    model = PetrinApplicationModel()
    state = PetrinLocalStateBuilder(model).build(model.theta0)

    projected = ProjectedInformationOperator(
        ridge=1e-8,
    ).fit(
        state.tractable_moments,
        state.demanding_moments,
        state.tractable_jacobian,
        state.demanding_jacobian,
    )

    gc = (
        state.tractable_moments
        - state.tractable_moments.mean(axis=0)
    )
    nuc = (
        projected.residual_moments
        - projected.residual_moments.mean(axis=0)
    )
    orthogonality = np.linalg.norm(
        nuc.T @ gc / state.n_markets
    )

    print("Petrin projected information")
    print(f"Projection B:          {projected.projection.shape}")
    print(
        f"Residual moments:      {projected.residual_moments.shape}"
    )
    print(
        f"Schur complement S:    {projected.schur_complement.shape}"
    )
    print(
        f"Residual Jacobian R:   {projected.residual_jacobian.shape}"
    )
    print(f"Information J:         {projected.information.shape}")
    print(f"Orthogonality norm:    {orthogonality:.6e}")
    print("Condition numbers:")
    for name, value in projected.condition_numbers.items():
        print(f"  {name:<20} {value:.6e}")
    print("Effective ranks:")
    for name, value in projected.effective_ranks.items():
        print(f"  {name:<20} {value}")


if __name__ == "__main__":
    main()

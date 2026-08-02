"""Build reduced-rank Petrin projected-information objects."""

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
        rank=3,
        ridge=1e-8,
    ).fit(
        state.tractable_moments,
        state.demanding_moments,
        state.tractable_jacobian,
        state.demanding_jacobian,
    )

    reduced = projected.basis_result.reduced_moments
    reduced_centered = reduced - reduced.mean(axis=0)
    residual_centered = (
        projected.residual_moments
        - projected.residual_moments.mean(axis=0)
    )
    orthogonality = np.linalg.norm(
        residual_centered.T
        @ reduced_centered
        / state.n_markets
    )

    basis = projected.basis_result

    print("Petrin reduced-rank projected information")
    print(f"Markets:                 {state.n_markets}")
    print(
        f"Original moments:        {basis.original_dimension}"
    )
    print(f"Empirical rank:          {basis.empirical_rank}")
    print(f"Retained rank:           {basis.retained_rank}")
    print(
        "Explained variance:      "
        f"{100 * basis.explained_variance_ratio:.6f}%"
    )
    print(
        "Discarded energy:        "
        f"{100 * basis.discarded_energy_ratio:.6f}%"
    )
    print(f"Projection B:            {projected.projection.shape}")
    print(
        f"Reduced projection:      {projected.reduced_projection.shape}"
    )
    print(
        f"Schur complement S:      {projected.schur_complement.shape}"
    )
    print(
        f"Residual Jacobian R:     {projected.residual_jacobian.shape}"
    )
    print(f"Information J:           {projected.information.shape}")
    print(f"Orthogonality norm:      {orthogonality:.6e}")
    print("Condition numbers:")
    for name, value in projected.condition_numbers.items():
        print(f"  {name:<20} {value:.6e}")


if __name__ == "__main__":
    main()

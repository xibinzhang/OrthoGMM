"""Audit Petrin projected-information diagnostics across ranks.

The expensive local Petrin state is built once. All rank calculations reuse
that same state and are therefore inexpensive.
"""

from __future__ import annotations

import csv
from pathlib import Path

from orthogmm.applications.petrin import (
    PetrinApplicationModel,
    PetrinLocalStateBuilder,
)
from orthogmm.diagnostics.rank_audit import RankAudit


def print_table(rows) -> None:
    header = (
        " r   Var.%    cond(Ogg)      cond(S)      cond(J)  "
        "rankJ  ridge(S)  ridge(J)      orthog"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row.rank:2d} "
            f"{100 * row.explained_variance_ratio:7.3f} "
            f"{row.condition_omega_gg:11.3e} "
            f"{row.condition_schur:11.3e} "
            f"{row.condition_information:11.3e} "
            f"{row.raw_rank_information:6d} "
            f"{row.ridge_schur:9.2e} "
            f"{row.ridge_information:9.2e} "
            f"{row.orthogonality_norm:11.3e}"
        )


def main() -> None:
    model = PetrinApplicationModel()
    state = PetrinLocalStateBuilder(model).build(model.theta0)

    rows = RankAudit(
        minimum_rank=1,
        maximum_rank=12,
        ridge=1e-8,
    ).run(
        state.tractable_moments,
        state.demanding_moments,
        state.tractable_jacobian,
        state.demanding_jacobian,
    )

    print()
    print("Petrin retained-rank audit")
    print(
        f"Markets={state.n_markets}, "
        f"tractable moments={state.n_tractable_moments}, "
        f"demanding moments={state.n_demanding_moments}, "
        f"parameters={state.parameter_dimension}"
    )
    print_table(rows)

    output = Path("results") / "petrin_rank_audit.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0].to_dict()),
        )
        writer.writeheader()
        writer.writerows(row.to_dict() for row in rows)

    print()
    print(f"Saved audit to {output}")


if __name__ == "__main__":
    main()

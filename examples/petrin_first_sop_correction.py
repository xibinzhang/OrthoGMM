"""Compute the first damped Petrin SOP correction at ranks 4 and 5.

The expensive local Petrin state is built once. Both rank specifications
reuse it. The script reports the proposed parameter steps but does not rerun
PyBLP at the updated parameters.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from orthogmm.applications.petrin import (
    PetrinApplicationModel,
    PetrinLocalStateBuilder,
)
from orthogmm.estimators.sequential import (
    SequentialProjectionCorrection,
)
from orthogmm.operators import ProjectedInformationOperator


def parameter_names() -> list[str]:
    return [
        "sigma_const",
        "sigma_hpwt",
        "sigma_space",
        "sigma_air",
        "sigma_mpd",
        "sigma_fwd",
        "pi_price_low_income",
        "pi_price_mid_income",
        "pi_price_high_income",
        "pi_mi_family_value",
        "pi_sw_family_value",
        "pi_su_family_value",
        "pi_pv_family_value",
    ]


def main() -> None:
    model = PetrinApplicationModel()
    state = PetrinLocalStateBuilder(model).build(model.theta0)

    correction = SequentialProjectionCorrection(
        damping=0.25,
        radius=0.20,
    )

    names = parameter_names()
    output_rows: list[dict[str, object]] = []

    print()
    print("First damped Petrin SOP correction")
    print(
        "Configuration: damping=0.25, Euclidean radius=0.20, "
        "ridge=1e-8"
    )

    for rank in (4, 5):
        projected = ProjectedInformationOperator(
            rank=rank,
            ridge=1e-8,
        ).fit(
            state.tractable_moments,
            state.demanding_moments,
            state.tractable_jacobian,
            state.demanding_jacobian,
        )
        result = correction.apply(state.theta, projected)

        print()
        print(f"Rank {rank}")
        print(
            f"  explained variance: "
            f"{100 * projected.basis_result.explained_variance_ratio:.6f}%"
        )
        print(
            f"  cond(J):            "
            f"{projected.condition_numbers['information']:.6e}"
        )
        print(f"  score norm:         {result.score_norm:.6e}")
        print(
            f"  raw direction norm: "
            f"{result.raw_direction_norm:.6e}"
        )
        print(
            f"  applied step norm:  "
            f"{result.applied_step_norm:.6e}"
        )
        print(f"  radius clipped:     {result.radius_clipped}")
        print()
        print(
            "  Parameter                         initial"
            "          step       updated"
        )
        print("  " + "-" * 71)

        for name, initial, step, updated in zip(
            names,
            result.theta_initial,
            result.applied_step,
            result.theta_updated,
            strict=True,
        ):
            print(
                f"  {name:<30} "
                f"{initial:12.6f} "
                f"{step:12.6f} "
                f"{updated:12.6f}"
            )
            output_rows.append(
                {
                    "rank": rank,
                    "parameter": name,
                    "initial": initial,
                    "raw_direction": result.raw_direction[
                        names.index(name)
                    ],
                    "applied_step": step,
                    "updated": updated,
                    "score_norm": result.score_norm,
                    "raw_direction_norm": result.raw_direction_norm,
                    "applied_step_norm": result.applied_step_norm,
                    "radius_clipped": result.radius_clipped,
                    "condition_information": (
                        projected.condition_numbers["information"]
                    ),
                    "explained_variance_ratio": (
                        projected.basis_result.explained_variance_ratio
                    ),
                }
            )

    output = Path("results") / "petrin_first_sop_correction.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(output_rows[0]),
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print()
    print(f"Saved correction audit to {output}")


if __name__ == "__main__":
    main()

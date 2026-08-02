"""Run aggregate-only Petrin localization.

This optimization excludes all micro moments. Its output is the preliminary
parameter vector at which the expensive local micro state should be built.
"""

from __future__ import annotations

import csv
from pathlib import Path

from orthogmm.applications.petrin import (
    PetrinApplicationModel,
    PetrinTractableLocalizer,
)


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
    result = PetrinTractableLocalizer(
        model,
        method="1s",
    ).fit()

    print()
    print("Petrin tractable localization")
    print(f"Converged:                 {result.converged}")
    print(f"Objective:                 {result.objective:.6e}")
    print(
        "Projected gradient norm:  "
        f"{result.projected_gradient_norm:.6e}"
    )
    print(
        f"Optimization iterations:   "
        f"{result.optimization_iterations}"
    )
    print(
        f"Objective evaluations:      "
        f"{result.objective_evaluations}"
    )
    print(
        f"Fixed-point iterations:     "
        f"{result.fixed_point_iterations}"
    )
    print(
        f"Contraction evaluations:    "
        f"{result.contraction_evaluations}"
    )
    print(f"Elapsed seconds:            {result.elapsed_seconds:.3f}")
    print(f"Update norm:                {result.update_norm:.6e}")
    print(
        f"Relative update norm:       "
        f"{result.relative_update_norm:.6e}"
    )
    print(f"Lower-bound hits:           {result.lower_bound_hits}")
    print(f"Upper-bound hits:           {result.upper_bound_hits}")

    print()
    print(
        "Parameter                         initial"
        "      localized         change"
    )
    print("-" * 72)

    rows = []
    for name, initial, localized, change in zip(
        parameter_names(),
        result.theta_initial,
        result.theta_localized,
        result.update,
        strict=True,
    ):
        print(
            f"{name:<30} "
            f"{initial:12.6f} "
            f"{localized:12.6f} "
            f"{change:12.6f}"
        )
        rows.append(
            {
                "parameter": name,
                "initial": initial,
                "localized": localized,
                "change": change,
            }
        )

    output = Path("results") / "petrin_localization.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Saved localization to {output}")


if __name__ == "__main__":
    main()

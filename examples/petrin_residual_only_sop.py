"""Run the matched-weight residual-only Petrin SOP correction.

This script uses the already-computed aggregate-only localizer in
``results/petrin_localization.csv``. It does not rerun nonlinear localization.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from orthogmm.applications.petrin import (
    PetrinApplicationModel,
    PetrinResidualOnlySOP,
)


LOCALIZATION_PATH = Path("results") / "petrin_localization.csv"
OUTPUT_PATH = Path("results") / "petrin_residual_only_sop.csv"


def load_localized_theta(
    path: Path,
) -> tuple[list[str], np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Localization file not found: {path}"
        )

    names: list[str] = []
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            names.append(row["parameter"])
            values.append(float(row["localized"]))

    theta = np.asarray(values, dtype=float)
    if theta.size != 13:
        raise ValueError(
            f"Expected 13 localized parameters, found {theta.size}."
        )
    if not np.all(np.isfinite(theta)):
        raise ValueError(
            "Localization file contains non-finite values."
        )
    return names, theta


def main() -> None:
    names, theta = load_localized_theta(LOCALIZATION_PATH)

    model = PetrinApplicationModel()
    result = PetrinResidualOnlySOP(
        model,
        ridge=1e-8,
        condition_limit=1e12,
        radius=1.0,
    ).fit(theta)

    projection = result.projected_information
    trust = result.trust_region

    print()
    print("Petrin matched-weight residual-only SOP")
    print("=" * 76)
    print(
        "Moment representation error:     "
        f"{result.moment_representation_error:.6e}"
    )
    print(
        "Tractable score norm:            "
        f"{result.tractable_score_norm:.6e}"
    )
    print(
        "Residual score norm:             "
        f"{result.residual_score_norm:.6e}"
    )
    print(
        "Full update norm:                "
        f"{result.full_update_norm:.6e}"
    )
    print(
        "Residual-only update norm:       "
        f"{result.residual_only_update_norm:.6e}"
    )
    print(
        "Update difference norm:          "
        f"{result.update_difference_norm:.6e}"
    )
    print(
        "Relative update difference:      "
        f"{result.relative_update_difference:.6e}"
    )
    print(
        "Applied trust-region step norm:  "
        f"{result.applied_step_norm:.6e}"
    )
    print(f"Trust region active:             {trust.active}")
    print(
        "Information condition number:    "
        f"{projection.condition_numbers['information']:.6e}"
    )
    print(
        "Information ridge:               "
        f"{projection.ridge_levels['information']:.6e}"
    )
    print(
        "Aggregate fixed-evaluation time: "
        f"{result.aggregate_elapsed_seconds:.3f}s"
    )
    print(
        "Local-state aggregate time:      "
        f"{result.local_state_aggregate_elapsed_seconds:.3f}s"
    )
    print(
        "Local demanding-block time:      "
        f"{result.micro_elapsed_seconds:.3f}s"
    )

    print()
    print(
        f"{'Parameter':<30}"
        f"{'Localized':>13}"
        f"{'SOP step':>13}"
        f"{'Updated':>13}"
    )
    print("-" * 69)

    rows: list[dict[str, object]] = []
    for index, name in enumerate(names):
        localized = result.theta_localized[index]
        step = result.applied_step[index]
        updated = result.theta_updated[index]

        print(
            f"{name:<30}"
            f"{localized:13.6f}"
            f"{step:13.6f}"
            f"{updated:13.6f}"
        )

        rows.append(
            {
                "parameter_index": index,
                "parameter": name,
                "localized": localized,
                "residual_only_step": step,
                "updated": updated,
                "tractable_score": result.tractable_score[index],
                "residual_score": result.residual_score[index],
                "projected_score": result.projected_score[index],
                "full_update": result.full_update[index],
                "residual_only_update": (
                    result.residual_only_update[index]
                ),
                "update_difference": result.update_difference[index],
                "tractable_score_norm": (
                    result.tractable_score_norm
                ),
                "residual_score_norm": result.residual_score_norm,
                "full_update_norm": result.full_update_norm,
                "residual_only_update_norm": (
                    result.residual_only_update_norm
                ),
                "update_difference_norm": (
                    result.update_difference_norm
                ),
                "relative_update_difference": (
                    result.relative_update_difference
                ),
                "applied_step_norm": result.applied_step_norm,
                "trust_region_active": trust.active,
                "trust_region_metric_norm": trust.metric_norm,
                "predicted_reduction": trust.predicted_reduction,
                "condition_information": (
                    projection.condition_numbers["information"]
                ),
                "ridge_omega_gg": (
                    projection.ridge_levels["omega_gg"]
                ),
                "ridge_residual_covariance": (
                    projection.ridge_levels[
                        "residual_covariance"
                    ]
                ),
                "ridge_information": (
                    projection.ridge_levels["information"]
                ),
                "moment_representation_error": (
                    result.moment_representation_error
                ),
                "aggregate_elapsed_seconds": (
                    result.aggregate_elapsed_seconds
                ),
                "local_state_aggregate_elapsed_seconds": (
                    result.local_state_aggregate_elapsed_seconds
                ),
                "micro_elapsed_seconds": (
                    result.micro_elapsed_seconds
                ),
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Saved residual-only SOP result to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

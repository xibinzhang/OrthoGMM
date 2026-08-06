"""Validate the matched-weight residual-only Petrin SOP update.

This script compares the saved aggregate-only localizer with the saved
residual-only SOP estimate under the same aggregate-plus-micro PyBLP
criterion. Both evaluations fix the nonlinear parameters, so no nonlinear
optimization is run.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from orthogmm.applications.petrin import PetrinApplicationModel
from orthogmm.applications.petrin.validation import PetrinSEIPValidator


LOCALIZATION_PATH = (
    Path("results") / "petrin_localization_matched_baseline.csv"
)
SOP_PATH = (
    Path("results") / "petrin_residual_only_sop_validated_bridge.csv"
)
OUTPUT_PATH = (
    Path("results") / "petrin_residual_only_validation.csv"
)


def load_localized(
    path: Path,
) -> tuple[list[str], np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Matched localization file not found: {path}"
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
            "Matched localization contains non-finite values."
        )
    return names, theta


def load_residual_only_sop(
    path: Path,
) -> tuple[list[str], np.ndarray, float]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Residual-only SOP file not found: {path}"
        )

    rows: list[tuple[int, str, float]] = []
    predicted_reduction = float("nan")

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            rows.append(
                (
                    int(row["parameter_index"]),
                    row["parameter"],
                    float(row["updated"]),
                )
            )
            if "predicted_reduction" in row:
                predicted_reduction = float(
                    row["predicted_reduction"]
                )

    rows.sort(key=lambda item: item[0])
    names = [name for _, name, _ in rows]
    theta = np.asarray([value for _, _, value in rows], dtype=float)

    if theta.size != 13:
        raise ValueError(
            f"Expected 13 residual-only SOP rows, found {theta.size}."
        )
    if not np.all(np.isfinite(theta)):
        raise ValueError(
            "Residual-only SOP estimate contains non-finite values."
        )
    return names, theta, predicted_reduction


def main() -> None:
    names, theta_localized = load_localized(LOCALIZATION_PATH)
    sop_names, theta_updated, predicted_reduction = (
        load_residual_only_sop(SOP_PATH)
    )

    if names != sop_names:
        raise ValueError(
            "Parameter ordering differs between localization and SOP files."
        )

    model = PetrinApplicationModel()
    result = PetrinSEIPValidator(
        model,
        include_micro=True,
        method="1s",
    ).compare(theta_localized, theta_updated)

    realized_to_predicted = float("nan")
    if np.isfinite(predicted_reduction) and predicted_reduction != 0:
        realized_to_predicted = (
            result.objective_improvement / predicted_reduction
        )

    print()
    print("Petrin residual-only SOP fixed-parameter validation")
    print("=" * 76)
    print(
        f"Localized objective:          "
        f"{result.localized.objective:.10e}"
    )
    print(
        f"Updated objective:            "
        f"{result.updated.objective:.10e}"
    )
    print(
        f"Objective change (new-old):   "
        f"{result.objective_change:.10e}"
    )
    print(
        f"Objective improvement:        "
        f"{result.objective_improvement:.10e}"
    )
    print(
        f"Relative improvement:         "
        f"{result.relative_objective_improvement:.10e}"
    )
    print(
        f"Objective improved:           "
        f"{result.objective_improved}"
    )
    print(
        f"Residual-only SOP step norm:  "
        f"{result.step_norm:.10e}"
    )
    print(
        f"Local predicted reduction:    "
        f"{predicted_reduction:.10e}"
    )
    print(
        f"Realized/predicted reduction: "
        f"{realized_to_predicted:.10e}"
    )
    print(
        f"Localized projected gradient: "
        f"{result.localized.projected_gradient_norm:.10e}"
    )
    print(
        f"Updated projected gradient:   "
        f"{result.updated.projected_gradient_norm:.10e}"
    )
    print(
        f"Localized elapsed seconds:    "
        f"{result.localized.elapsed_seconds:.3f}"
    )
    print(
        f"Updated elapsed seconds:      "
        f"{result.updated.elapsed_seconds:.3f}"
    )
    print(
        f"Localized fixed-point iters:  "
        f"{result.localized.fixed_point_iterations}"
    )
    print(
        f"Updated fixed-point iters:    "
        f"{result.updated.fixed_point_iterations}"
    )
    print(
        f"Localized contractions:       "
        f"{result.localized.contraction_evaluations}"
    )
    print(
        f"Updated contractions:         "
        f"{result.updated.contraction_evaluations}"
    )

    print()
    print(
        "Parameter                         localized"
        "       updated        change"
    )
    print("-" * 72)

    rows: list[dict[str, object]] = []
    for index, (name, localized, updated, change) in enumerate(
        zip(
            names,
            result.localized.theta,
            result.updated.theta,
            result.step,
            strict=True,
        )
    ):
        print(
            f"{name:<30} "
            f"{localized:12.6f} "
            f"{updated:12.6f} "
            f"{change:12.6f}"
        )
        rows.append(
            {
                "parameter_index": index,
                "parameter": name,
                "localized": localized,
                "updated": updated,
                "change": change,
                "localized_objective": result.localized.objective,
                "updated_objective": result.updated.objective,
                "objective_change": result.objective_change,
                "objective_improvement": (
                    result.objective_improvement
                ),
                "relative_objective_improvement": (
                    result.relative_objective_improvement
                ),
                "objective_improved": result.objective_improved,
                "predicted_reduction": predicted_reduction,
                "realized_to_predicted_reduction": (
                    realized_to_predicted
                ),
                "localized_projected_gradient_norm": (
                    result.localized.projected_gradient_norm
                ),
                "updated_projected_gradient_norm": (
                    result.updated.projected_gradient_norm
                ),
                "localized_elapsed_seconds": (
                    result.localized.elapsed_seconds
                ),
                "updated_elapsed_seconds": (
                    result.updated.elapsed_seconds
                ),
                "localized_fixed_point_iterations": (
                    result.localized.fixed_point_iterations
                ),
                "updated_fixed_point_iterations": (
                    result.updated.fixed_point_iterations
                ),
                "localized_contraction_evaluations": (
                    result.localized.contraction_evaluations
                ),
                "updated_contraction_evaluations": (
                    result.updated.contraction_evaluations
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
    print(f"Saved validation to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

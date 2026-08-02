"""Validate the baseline Petrin SEIP update with fixed PyBLP solves.

This script reads the saved micro-assisted localizer and integrated SEIP
output. It evaluates the full aggregate-plus-micro PyBLP criterion at the
localized vector and at the baseline rank-4, radius-1 SEIP vector. Nonlinear
parameters are fixed, so no optimizer is run.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from orthogmm.applications.petrin import PetrinApplicationModel
from orthogmm.applications.petrin.validation import PetrinSEIPValidator


LOCALIZATION_PATH = Path("results") / "petrin_micro_localization.csv"
SEIP_PATH = Path("results") / "petrin_seip_estimate.csv"
OUTPUT_PATH = Path("results") / "petrin_seip_validation.csv"


def load_localized(path: Path) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    values: list[float] = []

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            names.append(row["parameter"])
            values.append(float(row["localized"]))

    theta = np.asarray(values, dtype=float)
    if theta.size != 13:
        raise ValueError("Expected 13 localized parameters.")
    return names, theta


def load_baseline_seip(path: Path) -> np.ndarray:
    rows: list[tuple[int, float]] = []

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if row["configuration"] == "baseline":
                rows.append(
                    (
                        int(row["parameter_index"]),
                        float(row["updated"]),
                    )
                )

    rows.sort(key=lambda item: item[0])
    theta = np.asarray([value for _, value in rows], dtype=float)
    if theta.size != 13:
        raise ValueError(
            "Expected 13 baseline SEIP parameter rows."
        )
    return theta


def main() -> None:
    names, theta_localized = load_localized(LOCALIZATION_PATH)
    theta_updated = load_baseline_seip(SEIP_PATH)

    model = PetrinApplicationModel()
    result = PetrinSEIPValidator(
        model,
        include_micro=True,
        method="1s",
    ).compare(theta_localized, theta_updated)

    print()
    print("Petrin SEIP fixed-parameter validation")
    print(f"Localized objective:       {result.localized.objective:.6e}")
    print(f"Updated objective:         {result.updated.objective:.6e}")
    print(f"Objective change:          {result.objective_change:.6e}")
    print(f"Objective improvement:     {result.objective_improvement:.6e}")
    print(f"Objective improved:        {result.objective_improved}")
    print(f"SEIP step norm:            {result.step_norm:.6e}")
    print(
        f"Localized elapsed seconds: {result.localized.elapsed_seconds:.3f}"
    )
    print(
        f"Updated elapsed seconds:   {result.updated.elapsed_seconds:.3f}"
    )
    print(
        "Localized contractions:   "
        f"{result.localized.contraction_evaluations}"
    )
    print(
        "Updated contractions:     "
        f"{result.updated.contraction_evaluations}"
    )

    print()
    print(
        "Parameter                         localized"
        "       updated        change"
    )
    print("-" * 72)

    rows = []
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
                "objective_improvement": result.objective_improvement,
                "objective_improved": result.objective_improved,
                "localized_elapsed_seconds": (
                    result.localized.elapsed_seconds
                ),
                "updated_elapsed_seconds": (
                    result.updated.elapsed_seconds
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

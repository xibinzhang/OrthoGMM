"""Run the integrated Petrin SEIP estimator from saved localization."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from orthogmm.applications.petrin import (
    PetrinApplicationModel,
    PetrinLocalStateBuilder,
)
from orthogmm.estimators.seip import (
    SequentialEfficientInfluenceProjection,
)


LOCALIZATION_PATH = Path("results") / "petrin_micro_localization.csv"
OUTPUT_PATH = Path("results") / "petrin_seip_estimate.csv"


def load_localized_theta(
    path: Path,
) -> tuple[list[str], np.ndarray]:
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


def main() -> None:
    names, theta = load_localized_theta(LOCALIZATION_PATH)

    model = PetrinApplicationModel()
    state = PetrinLocalStateBuilder(model).build(theta)

    configurations = (
        ("baseline", 4, 1.0),
        ("rank_sensitivity", 5, 1.0),
        ("radius_sensitivity", 4, float(np.sqrt(theta.size))),
    )

    rows: list[dict[str, object]] = []

    for label, rank, radius in configurations:
        result = SequentialEfficientInfluenceProjection(
            rank=rank,
            ridge=1e-8,
            metric_type="information",
            radius=radius,
        ).fit(
            theta,
            state.tractable_moments,
            state.demanding_moments,
            state.tractable_jacobian,
            state.demanding_jacobian,
        )

        projected = result.projected_information
        trust = result.trust_region

        print()
        print(f"Petrin integrated SEIP: {label}")
        print(f"Retained rank:          {result.retained_rank}")
        print(
            f"Explained variance:     "
            f"{100 * result.explained_variance_ratio:.6f}%"
        )
        print(
            f"cond(J):                "
            f"{projected.condition_numbers['information']:.6e}"
        )
        print(f"Information radius:     {trust.radius:.6e}")
        print(f"Metric step norm:       {trust.metric_norm:.6e}")
        print(f"Euclidean step norm:    {result.step_norm:.6e}")
        print(f"Relative step norm:     {result.relative_step_norm:.6e}")
        print(f"Score norm:             {result.score_norm:.6e}")
        print(f"Trust region active:    {trust.active}")
        print(
            f"Predicted reduction:    "
            f"{trust.predicted_reduction:.6e}"
        )
        print(
            "Parameter                         localized"
            "          step       updated"
        )
        print("-" * 72)

        for index, (name, localized, step, updated) in enumerate(
            zip(
                names,
                result.theta_initial,
                result.step,
                result.theta_updated,
                strict=True,
            )
        ):
            print(
                f"{name:<30} "
                f"{localized:12.6f} "
                f"{step:12.6f} "
                f"{updated:12.6f}"
            )
            rows.append(
                {
                    "configuration": label,
                    "rank": rank,
                    "radius": radius,
                    "parameter_index": index,
                    "parameter": name,
                    "localized": localized,
                    "step": step,
                    "updated": updated,
                    "metric_norm": trust.metric_norm,
                    "euclidean_step_norm": result.step_norm,
                    "relative_step_norm": result.relative_step_norm,
                    "score_norm": result.score_norm,
                    "predicted_reduction": trust.predicted_reduction,
                    "trust_region_active": trust.active,
                    "condition_information": (
                        projected.condition_numbers["information"]
                    ),
                    "explained_variance_ratio": (
                        result.explained_variance_ratio
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
    print(f"Saved integrated SEIP estimates to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

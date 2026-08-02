"""Compare scale-aware Petrin SOP trust-region corrections.

The script loads the saved micro-assisted localization, rebuilds the local
state once, and compares parameter-scale and information-metric trust regions
for retained ranks 4 and 5.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from orthogmm.applications.petrin import (
    PetrinApplicationModel,
    PetrinLocalStateBuilder,
)
from orthogmm.linalg import solve
from orthogmm.optimization import QuadraticTrustRegion
from orthogmm.operators import ProjectedInformationOperator


LOCALIZATION_PATH = Path("results") / "petrin_micro_localization.csv"
OUTPUT_PATH = Path("results") / "petrin_trust_region_corrections.csv"


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


def projected_score(projected) -> np.ndarray:
    basis = projected.basis_result
    g_bar = basis.reduced_moments.mean(axis=0)
    nu_bar = projected.residual_moments.mean(axis=0)

    tractable = basis.reduced_jacobian.T @ solve(
        projected.omega_gg,
        g_bar,
    )
    demanding = projected.residual_jacobian.T @ solve(
        projected.schur_complement,
        nu_bar,
    )
    return tractable + demanding


def main() -> None:
    names, theta = load_localized_theta(LOCALIZATION_PATH)

    model = PetrinApplicationModel()
    state = PetrinLocalStateBuilder(model).build(theta)

    solvers = {
        "parameter_scale": QuadraticTrustRegion(
            radius=0.20,
            metric_type="parameter_scale",
            ridge=1e-8,
        ),
        "information": QuadraticTrustRegion(
            radius=1.00,
            metric_type="information",
            ridge=1e-8,
        ),
    }

    rows: list[dict[str, object]] = []

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
        score = projected_score(projected)

        for metric_name, solver in solvers.items():
            result = solver.solve(
                projected.information,
                score,
                theta=theta,
            )
            updated = theta + result.step
            scales = np.maximum(1.0, np.abs(theta))
            relative_step = result.step / scales

            print()
            print(
                f"Petrin trust-region correction: "
                f"rank={rank}, metric={metric_name}"
            )
            print(
                f"cond(J):                 "
                f"{projected.condition_numbers['information']:.6e}"
            )
            print(f"score norm:              {np.linalg.norm(score):.6e}")
            print(
                f"unconstrained norm:      "
                f"{result.unconstrained_euclidean_norm:.6e}"
            )
            print(f"metric norm:             {result.metric_norm:.6e}")
            print(
                f"Euclidean step norm:     "
                f"{result.euclidean_norm:.6e}"
            )
            print(
                f"multiplier:              "
                f"{result.lagrange_multiplier:.6e}"
            )
            print(f"active:                  {result.active}")
            print(f"solver converged:        {result.converged}")
            print(
                f"predicted reduction:     "
                f"{result.predicted_reduction:.6e}"
            )
            print(
                "Parameter                         localized"
                "          step    relative       updated"
            )
            print("-" * 86)

            for index, (
                name,
                localized,
                step,
                relative,
                final,
            ) in enumerate(
                zip(
                    names,
                    theta,
                    result.step,
                    relative_step,
                    updated,
                    strict=True,
                )
            ):
                print(
                    f"{name:<30} "
                    f"{localized:12.6f} "
                    f"{step:12.6f} "
                    f"{relative:11.6f} "
                    f"{final:12.6f}"
                )
                rows.append(
                    {
                        "rank": rank,
                        "metric": metric_name,
                        "parameter_index": index,
                        "parameter": name,
                        "localized": localized,
                        "step": step,
                        "relative_step": relative,
                        "updated": final,
                        "metric_norm": result.metric_norm,
                        "euclidean_step_norm": result.euclidean_norm,
                        "unconstrained_norm": (
                            result.unconstrained_euclidean_norm
                        ),
                        "lagrange_multiplier": (
                            result.lagrange_multiplier
                        ),
                        "active": result.active,
                        "predicted_reduction": (
                            result.predicted_reduction
                        ),
                        "condition_information": (
                            projected.condition_numbers["information"]
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
    print(f"Saved trust-region corrections to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

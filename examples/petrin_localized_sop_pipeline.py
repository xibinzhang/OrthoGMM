"""Build the Petrin local SOP state from a saved micro localization.

This script does not rerun the 20-minute localization. It reads
``results/petrin_micro_localization.csv``, builds the expensive local state
once, audits ranks 1--12, and computes damped SOP corrections at ranks 4 and 5.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from orthogmm.applications.petrin import (
    PetrinApplicationModel,
    PetrinLocalStateBuilder,
)
from orthogmm.diagnostics.rank_audit import RankAudit
from orthogmm.estimators.sequential import SequentialProjectionCorrection
from orthogmm.operators import ProjectedInformationOperator


LOCALIZATION_PATH = Path("results") / "petrin_micro_localization.csv"
RANK_OUTPUT = Path("results") / "petrin_localized_rank_audit.csv"
CORRECTION_OUTPUT = Path("results") / "petrin_localized_sop_correction.csv"


def load_localized_theta(path: Path) -> tuple[list[str], np.ndarray]:
    """Load parameter names and localized values from the saved CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"Localization file not found: {path}. "
            "Run petrin_micro_localization_smoke.py first."
        )

    names: list[str] = []
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"parameter", "localized"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Localization CSV must contain parameter and localized columns."
            )
        for row in reader:
            names.append(row["parameter"])
            values.append(float(row["localized"]))

    theta = np.asarray(values, dtype=float)
    if theta.size != 13:
        raise ValueError(
            f"Expected 13 localized parameters, found {theta.size}."
        )
    if not np.all(np.isfinite(theta)):
        raise ValueError("Localization CSV contains non-finite values.")
    return names, theta


def write_rank_audit(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0].to_dict()),
        )
        writer.writeheader()
        writer.writerows(row.to_dict() for row in rows)


def main() -> None:
    names, theta_localized = load_localized_theta(LOCALIZATION_PATH)

    print("Loaded localized Petrin vector:")
    print(theta_localized)

    model = PetrinApplicationModel()
    state = PetrinLocalStateBuilder(model).build(theta_localized)

    print()
    print("Localized Petrin local state")
    print(f"Markets:                 {state.n_markets}")
    print(f"Tractable moments:       {state.tractable_moments.shape}")
    print(f"Demanding moments:       {state.demanding_moments.shape}")
    print(f"Tractable Jacobian:      {state.tractable_jacobian.shape}")
    print(f"Demanding Jacobian:      {state.demanding_jacobian.shape}")
    print(
        "Demanding mean norm:    "
        f"{np.linalg.norm(state.demanding_moments.mean(axis=0)):.6e}"
    )

    rank_rows = RankAudit(
        minimum_rank=1,
        maximum_rank=12,
        ridge=1e-8,
    ).run(
        state.tractable_moments,
        state.demanding_moments,
        state.tractable_jacobian,
        state.demanding_jacobian,
    )
    write_rank_audit(rank_rows, RANK_OUTPUT)

    print()
    print(
        " r   Var.%    cond(Ogg)      cond(S)      cond(J)  "
        "rankJ  ridge(S)  ridge(J)      orthog"
    )
    print("-" * 104)
    for row in rank_rows:
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

    correction = SequentialProjectionCorrection(
        damping=0.25,
        radius=0.20,
    )
    output_rows: list[dict[str, object]] = []

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
        result = correction.apply(theta_localized, projected)

        print()
        print(f"Localized SOP correction: rank {rank}")
        print(
            f"Explained variance:       "
            f"{100 * projected.basis_result.explained_variance_ratio:.6f}%"
        )
        print(
            f"Information condition:    "
            f"{projected.condition_numbers['information']:.6e}"
        )
        print(f"Score norm:               {result.score_norm:.6e}")
        print(
            f"Raw direction norm:       "
            f"{result.raw_direction_norm:.6e}"
        )
        print(
            f"Applied step norm:        "
            f"{result.applied_step_norm:.6e}"
        )
        print(f"Radius clipped:           {result.radius_clipped}")
        print(
            "Parameter                         localized"
            "          step       updated"
        )
        print("-" * 72)

        for index, (name, localized, raw, step, updated) in enumerate(
            zip(
                names,
                result.theta_initial,
                result.raw_direction,
                result.applied_step,
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
            output_rows.append(
                {
                    "rank": rank,
                    "parameter_index": index,
                    "parameter": name,
                    "localized": localized,
                    "raw_direction": raw,
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

    CORRECTION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with CORRECTION_OUTPUT.open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(output_rows[0]),
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print()
    print(f"Saved rank audit to {RANK_OUTPUT}")
    print(f"Saved SOP corrections to {CORRECTION_OUTPUT}")


if __name__ == "__main__":
    main()

"""Paired diagnostics for Section 5 Full GMM and residual-only SOP."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "raw_csv",
        type=Path,
        help="Raw CSV produced by examples/section5_run.py.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return float("nan")
    return float(value)


def percentile(values: np.ndarray, probability: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, probability))


def main() -> None:
    args = parse_args()
    rows = read_rows(args.raw_csv)

    cells = sorted({
        (
            int(row["n"]),
            int(row["basis_k"]),
            int(row["q_nodes"]),
        )
        for row in rows
    })

    print("\nPaired Full--SOP validation")
    print("=" * 132)
    print(
        " n   K   Q   pairs  mean |theta diff|  p95 |theta diff|  "
        "mean |SE diff|  speedup  eval reduction"
    )
    print("-" * 132)

    for n, basis_k, q_nodes in cells:
        cell = [
            row
            for row in rows
            if (
                int(row["n"]) == n
                and int(row["basis_k"]) == basis_k
                and int(row["q_nodes"]) == q_nodes
                and row.get("success", "").lower() == "true"
            )
        ]
        by_key = {
            (int(row["replication"]), row["estimator"]): row
            for row in cell
        }
        replications = sorted({
            int(row["replication"])
            for row in cell
            if (
                (int(row["replication"]), "Full") in by_key
                and (int(row["replication"]), "SOP") in by_key
            )
        })

        theta_distances = []
        se_distances = []
        speedups = []
        reductions = []

        for replication in replications:
            full = by_key[(replication, "Full")]
            sop = by_key[(replication, "SOP")]

            full_theta = np.array([
                number(full, "alpha"),
                number(full, "beta"),
            ])
            sop_theta = np.array([
                number(sop, "alpha"),
                number(sop, "beta"),
            ])
            full_se = np.array([
                number(full, "se_alpha"),
                number(full, "se_beta"),
            ])
            sop_se = np.array([
                number(sop, "se_alpha"),
                number(sop, "se_beta"),
            ])

            theta_distances.append(
                float(np.linalg.norm(full_theta - sop_theta))
            )
            se_distances.append(
                float(np.linalg.norm(full_se - sop_se))
            )
            speedups.append(
                number(full, "wall_time_seconds")
                / number(sop, "wall_time_seconds")
            )
            reductions.append(
                1.0
                - number(sop, "demanding_evaluations")
                / number(full, "demanding_evaluations")
            )

        theta_array = np.asarray(theta_distances)
        se_array = np.asarray(se_distances)

        print(
            f"{n:4d} {basis_k:3d} {q_nodes:3d} "
            f"{len(replications):7d} "
            f"{np.mean(theta_array):18.6g} "
            f"{percentile(theta_array, 0.95):17.6g} "
            f"{np.mean(se_array):15.6g} "
            f"{np.mean(speedups):8.3f} "
            f"{100.0 * np.mean(reductions):13.3f}%"
        )

    sop_rows = [
        row
        for row in rows
        if (
            row["estimator"] == "SOP"
            and row.get("success", "").lower() == "true"
        )
    ]

    print("\nResidual-only SOP diagnostics")
    print("=" * 112)
    print(
        " n   K   Q   runs   mean FOC norm   max FOC norm   "
        "mean update diff   max update diff"
    )
    print("-" * 112)

    for n, basis_k, q_nodes in cells:
        cell = [
            row
            for row in sop_rows
            if (
                int(row["n"]) == n
                and int(row["basis_k"]) == basis_k
                and int(row["q_nodes"]) == q_nodes
            )
        ]
        foc = np.asarray([
            number(row, "tractable_foc_norm")
            for row in cell
        ])
        update_diff = np.asarray([
            number(row, "update_difference_norm")
            for row in cell
        ])
        foc = foc[np.isfinite(foc)]
        update_diff = update_diff[np.isfinite(update_diff)]

        print(
            f"{n:4d} {basis_k:3d} {q_nodes:3d} "
            f"{len(cell):6d} "
            f"{np.mean(foc):15.6g} "
            f"{np.max(foc):14.6g} "
            f"{np.mean(update_diff):18.6g} "
            f"{np.max(update_diff):17.6g}"
        )


if __name__ == "__main__":
    main()

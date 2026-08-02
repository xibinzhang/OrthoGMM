r"""Command-line driver for Section 5 Monte Carlo experiments.

Examples
--------
Baseline experiment:
    py examples\section5_run.py --experiment baseline --replications 200

Basis-complexity experiment:
    py examples\section5_run.py --experiment basis --replications 200

Quadrature-complexity experiment:
    py examples\section5_run.py --experiment quadrature --replications 200
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np

from orthogmm import (
    RandomCoefficientIntegration,
    RandomCoefficientMomentModel,
    fit_full_gmm,
    fit_seip,
    fit_tractable_gmm,
)
from orthogmm.simulation import (
    Experiment,
    NonlinearRandomCoefficientDesign,
)


THETA0 = np.array([0.0, 0.0])
BOUNDS = [(-5.0, 5.0), (-5.0, 5.0)]
MASTER_SEED = 19890604


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Section 5 Monte Carlo experiments."
    )
    parser.add_argument(
        "--experiment",
        choices=("baseline", "basis", "quadrature"),
        default="baseline",
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
    )
    return parser.parse_args()


def make_model(data, *, basis_k: int, q_nodes: int):
    return RandomCoefficientMomentModel.from_data(
        data,
        basis_k=basis_k,
        basis_family="fourier",
        integration=RandomCoefficientIntegration(
            mode="quadrature",
            q_nodes=q_nodes,
        ),
    )


def make_estimators(*, basis_k: int, q_nodes: int):
    def initial(data):
        return fit_tractable_gmm(
            make_model(data, basis_k=basis_k, q_nodes=q_nodes),
            theta0=THETA0,
            bounds=BOUNDS,
        )

    def full(data):
        return fit_full_gmm(
            make_model(data, basis_k=basis_k, q_nodes=q_nodes),
            theta0=THETA0,
            bounds=BOUNDS,
        )

    def sop(data):
        return fit_seip(
            make_model(data, basis_k=basis_k, q_nodes=q_nodes),
            theta0=THETA0,
            bounds=BOUNDS,
            ridge=1e-8,
        )

    return {
        "Initial": initial,
        "Full": full,
        "SOP": sop,
    }


def save_rows(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_cell(
    *,
    n: int,
    replications: int,
    basis_k: int,
    q_nodes: int,
    seed: int,
):
    design = NonlinearRandomCoefficientDesign(
        n=n,
        alpha=0.0,
        beta=1.0,
        lambda_=0.5,
        sigma=0.8,
    )

    experiment = Experiment(
        design=design,
        estimators=make_estimators(
            basis_k=basis_k,
            q_nodes=q_nodes,
        ),
        repetitions=replications,
        seed=seed,
        continue_on_error=True,
    )

    results = experiment.run()
    summaries = results.summarize(confidence_level=0.95)

    rows = []
    for summary in summaries:
        row = asdict(summary)
        row.update(
            {
                "n": n,
                "basis_k": basis_k,
                "q_nodes": q_nodes,
                "failed_runs": results.n_failures,
            }
        )
        rows.append(row)

    return rows


def run_baseline(replications: int):
    rows = []
    for index, n in enumerate((250, 500, 1000, 2000)):
        print(f"Baseline: n={n}, R={replications}")
        rows.extend(
            run_cell(
                n=n,
                replications=replications,
                basis_k=1,
                q_nodes=40,
                seed=MASTER_SEED + index,
            )
        )
    return rows


def run_basis(replications: int):
    rows = []
    n = 1000
    for index, basis_k in enumerate((1, 2, 4, 8)):
        print(f"Basis complexity: K={basis_k}, n={n}, R={replications}")
        rows.extend(
            run_cell(
                n=n,
                replications=replications,
                basis_k=basis_k,
                q_nodes=40,
                seed=MASTER_SEED + 100 + index,
            )
        )
    return rows


def run_quadrature(replications: int):
    rows = []
    n = 1000
    for index, q_nodes in enumerate((10, 20, 40, 80)):
        print(f"Quadrature complexity: Q={q_nodes}, n={n}, R={replications}")
        rows.extend(
            run_cell(
                n=n,
                replications=replications,
                basis_k=1,
                q_nodes=q_nodes,
                seed=MASTER_SEED + 200 + index,
            )
        )
    return rows


def main() -> None:
    args = parse_args()

    if args.replications <= 0:
        raise ValueError("replications must be positive.")

    if args.experiment == "baseline":
        rows = run_baseline(args.replications)
    elif args.experiment == "basis":
        rows = run_basis(args.replications)
    else:
        rows = run_quadrature(args.replications)

    output_path = (
        args.output_dir
        / f"section5_{args.experiment}_R{args.replications}.csv"
    )
    save_rows(rows, output_path)

    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()

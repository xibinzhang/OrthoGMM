r"""Twenty-replication validation of the Section 5 Monte Carlo design.

Run from the repository root with:

    py examples\section5_monte_carlo.py
"""

from __future__ import annotations

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


REPETITIONS = 20
SAMPLE_SIZE = 500
MASTER_SEED = 19890604
THETA0 = np.array([0.0, 0.0])
BOUNDS = [(-5.0, 5.0), (-5.0, 5.0)]


def make_model(data):
    """Construct the Section 5 moment model for one replication."""

    return RandomCoefficientMomentModel.from_data(
        data,
        basis_k=1,
        basis_family="fourier",
        integration=RandomCoefficientIntegration(
            mode="quadrature",
            q_nodes=40,
        ),
    )


def run_initial(data):
    """Run the tractable preliminary estimator."""

    return fit_tractable_gmm(
        make_model(data),
        theta0=THETA0,
        bounds=BOUNDS,
    )


def run_full(data):
    """Run full-system GMM with repeated demanding evaluations."""

    return fit_full_gmm(
        make_model(data),
        theta0=THETA0,
        bounds=BOUNDS,
    )


def run_sop(data):
    """Run sequential orthogonal projection estimation."""

    return fit_seip(
        make_model(data),
        theta0=THETA0,
        bounds=BOUNDS,
        ridge=1e-8,
    )


def format_optional(value, digits=4):
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def print_summary(summaries):
    """Print one row per estimator and parameter."""

    header = (
        f"{'Estimator':<10}"
        f"{'Parameter':<11}"
        f"{'True':>8}"
        f"{'Bias':>10}"
        f"{'RMSE':>10}"
        f"{'Emp SD':>10}"
        f"{'Mean SE':>10}"
        f"{'Cover':>9}"
        f"{'Success':>10}"
        f"{'Time(s)':>10}"
        f"{'Demand':>10}"
    )
    print(header)
    print("-" * len(header))

    parameter_names = {0: "alpha", 1: "beta"}

    for row in summaries:
        print(
            f"{row.estimator:<10}"
            f"{parameter_names.get(row.parameter_index, str(row.parameter_index)):<11}"
            f"{row.true_value:>8.3f}"
            f"{row.bias:>10.4f}"
            f"{row.rmse:>10.4f}"
            f"{row.empirical_sd:>10.4f}"
            f"{format_optional(row.mean_standard_error):>10}"
            f"{format_optional(row.coverage, 3):>9}"
            f"{row.success_rate:>10.3f}"
            f"{row.mean_runtime_seconds:>10.4f}"
            f"{format_optional(row.mean_demanding_evaluations, 1):>10}"
        )


def save_summary(summaries, output_path):
    """Save the Monte Carlo summary as CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(asdict(summaries[0]).keys()),
        )
        writer.writeheader()

        for row in summaries:
            writer.writerow(asdict(row))


def main():
    design = NonlinearRandomCoefficientDesign(
        n=SAMPLE_SIZE,
        alpha=0.0,
        beta=1.0,
        lambda_=0.5,
        sigma=0.8,
    )

    experiment = Experiment(
        design=design,
        estimators={
            "Initial": run_initial,
            "Full": run_full,
            "SOP": run_sop,
        },
        repetitions=REPETITIONS,
        seed=MASTER_SEED,
        continue_on_error=True,
    )

    print(
        "Running Section 5 Monte Carlo validation "
        f"with R={REPETITIONS}, n={SAMPLE_SIZE}..."
    )

    results = experiment.run()
    summaries = results.summarize(
        confidence_level=0.95,
    )

    print()
    print_summary(summaries)

    output_path = Path(
        "results/section5_monte_carlo_smoke.csv"
    )
    save_summary(summaries, output_path)

    print(
        f"\nSaved summary to {output_path}"
    )
    print(
        f"Failed estimator runs: {results.n_failures}"
    )


if __name__ == "__main__":
    main()

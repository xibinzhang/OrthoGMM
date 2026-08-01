r"""Small end-to-end demonstration of the Section 5 design.

Run from the repository root with:

    py examples\section5_smoke.py
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from orthogmm import (
    RandomCoefficientIntegration,
    RandomCoefficientMomentModel,
    fit_full_gmm,
    fit_seip,
    fit_tractable_gmm,
)
from orthogmm.simulation import NonlinearRandomCoefficientDesign


def _fit_with_wall_time(function, *args, **kwargs):
    start = perf_counter()
    result = function(*args, **kwargs)
    return result, perf_counter() - start


def _print_results(rows) -> None:
    header = (
        f"{'Estimator':<14}"
        f"{'alpha':>11}"
        f"{'beta':>11}"
        f"{'SE(alpha)':>12}"
        f"{'SE(beta)':>12}"
        f"{'time (s)':>11}"
        f"{'demanding':>12}"
        f"{'success':>10}"
    )
    print(header)
    print("-" * len(header))

    for name, result, wall_time in rows:
        demanding = result.counts.demanding_moments_total
        print(
            f"{name:<14}"
            f"{result.theta[0]:>11.6f}"
            f"{result.theta[1]:>11.6f}"
            f"{result.standard_errors[0]:>12.6f}"
            f"{result.standard_errors[1]:>12.6f}"
            f"{wall_time:>11.4f}"
            f"{demanding:>12d}"
            f"{str(result.success):>10}"
        )

        if result.warnings:
            print(" " * 4 + "Warnings: " + "; ".join(result.warnings))


def main() -> None:
    design = NonlinearRandomCoefficientDesign(
        n=500,
        alpha=0.0,
        beta=1.0,
        lambda_=0.5,
        sigma=0.8,
        seed=19890604,
    )
    data = design.generate()

    model = RandomCoefficientMomentModel.from_data(
        data,
        basis_k=1,
        basis_family="fourier",
        integration=RandomCoefficientIntegration(
            mode="quadrature",
            q_nodes=40,
        ),
    )

    theta0 = np.array([0.0, 0.0])

    initial, initial_time = _fit_with_wall_time(
        fit_tractable_gmm,
        model,
        theta0,
    )

    full, full_time = _fit_with_wall_time(
        fit_full_gmm,
        model,
        initial.theta,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
    )

    sop, sop_time = _fit_with_wall_time(
        fit_seip,
        model,
        theta0,
        preliminary_theta=initial.theta,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        ridge=1e-8,
    )

    print("\nSection 5 nonlinear random-coefficient smoke experiment")
    print(
        "Truth: "
        f"alpha={data['theta_true'][0]:.6f}, "
        f"beta={data['theta_true'][1]:.6f}\n"
    )

    _print_results(
        [
            ("Initial", initial, initial_time),
            ("Full", full, full_time),
            ("SOP", sop, sop_time),
        ]
    )

    distance = np.linalg.norm(full.theta - sop.theta)
    print(
        "\nEuclidean distance between Full and SOP estimates: "
        f"{distance:.6g}"
    )


if __name__ == "__main__":
    main()

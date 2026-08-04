"""Demanding-moment scaling experiment for OrthoGMM Version 1.0.

The experiment holds the sample size and aggregate instrument relevance fixed
while varying the number of computationally demanding moments:

    K_h = 1, 2, 4, 8.

The tractable subsystem always contains:

    1. an intercept moment;
    2. an exogenous-regressor moment;
    3. one excluded-instrument moment.

The demanding subsystem contains K_h additional excluded-instrument moments.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from orthogmm import (
    GridBenchmark,
    fit_full,
    fit_projection,
    fit_tractable,
)


@dataclass
class DemandingMomentLinearIVDesign:
    """Linear IV design with a fixed tractable block and scalable demanding block.

    The first excluded instrument belongs to the tractable subsystem. Additional
    excluded instruments belong to the demanding subsystem.

    The demanding instruments enter the first stage through a normalised sum,
    so their aggregate relevance remains approximately constant as their number
    increases.
    """

    n: int = 500
    demanding_moments: int = 1
    beta: tuple[float, float] = (1.0, -0.5)
    endogeneity: float = 0.5
    tractable_strength: float = 0.8
    demanding_strength: float = 0.8
    exogenous_strength: float = 0.5
    sigma: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError("n must be positive.")

        if self.demanding_moments <= 0:
            raise ValueError("demanding_moments must be positive.")

        if len(self.beta) != 2:
            raise ValueError("beta must contain exactly two coefficients.")

        if not -0.999 < self.endogeneity < 0.999:
            raise ValueError(
                "endogeneity must lie strictly between -0.999 and 0.999."
            )

        if self.tractable_strength <= 0:
            raise ValueError("tractable_strength must be positive.")

        if self.demanding_strength <= 0:
            raise ValueError("demanding_strength must be positive.")

        if self.sigma <= 0:
            raise ValueError("sigma must be positive.")

    def generate(self, seed: int | None = None) -> dict[str, Any]:
        """Generate one Monte Carlo sample."""

        effective_seed = self.seed if seed is None else seed
        rng = np.random.default_rng(effective_seed)

        exogenous_regressor = rng.normal(size=self.n)

        tractable_instrument = rng.normal(size=self.n)
        demanding_instruments = rng.normal(
            size=(self.n, self.demanding_moments)
        )

        structural_shock = rng.normal(size=self.n)
        independent_shock = rng.normal(size=self.n)

        endogenous_disturbance = (
            self.endogeneity * structural_shock
            + np.sqrt(1.0 - self.endogeneity**2) * independent_shock
        )

        # This normalisation keeps Var(demanding_index) approximately equal
        # to one for every value of demanding_moments.
        demanding_index = (
            demanding_instruments.sum(axis=1)
            / np.sqrt(float(self.demanding_moments))
        )

        endogenous_regressor = (
            self.tractable_strength * tractable_instrument
            + self.demanding_strength * demanding_index
            + self.exogenous_strength * exogenous_regressor
            + endogenous_disturbance
        )

        X = np.column_stack(
            (
                exogenous_regressor,
                endogenous_regressor,
            )
        ).astype(float)

        Z = np.column_stack(
            (
                np.ones(self.n),
                exogenous_regressor,
                tractable_instrument,
                demanding_instruments,
            )
        ).astype(float)

        theta_true = np.asarray(self.beta, dtype=float)
        structural_error = self.sigma * structural_shock
        y = X @ theta_true + structural_error

        return {
            "y": y.astype(float),
            "X": X,
            "Z": Z,
            "theta_true": theta_true,
            "n": self.n,
            "demanding_moments": self.demanding_moments,
            "seed": effective_seed,
        }


class LinearMomentModel:
    """Linear IV moments split into tractable and demanding blocks."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.y = np.asarray(data["y"], dtype=float)
        self.x = np.asarray(data["X"], dtype=float)
        self.z = np.asarray(data["Z"], dtype=float)

        if self.y.ndim != 1:
            raise ValueError("data['y'] must be one-dimensional.")

        if self.x.ndim != 2:
            raise ValueError("data['X'] must be two-dimensional.")

        if self.z.ndim != 2:
            raise ValueError("data['Z'] must be two-dimensional.")

        if not (
            self.y.shape[0]
            == self.x.shape[0]
            == self.z.shape[0]
        ):
            raise ValueError(
                "y, X, and Z must contain the same number of observations."
            )

        if self.z.shape[1] < 4:
            raise ValueError(
                "The design must contain at least one demanding moment."
            )

    def tractable_moments(self, theta: np.ndarray) -> np.ndarray:
        """Return the three identified tractable moments."""

        residual = self.y - self.x @ theta

        # Intercept, exogenous regressor, and one excluded instrument.
        return self.z[:, :3] * residual[:, None]

    def demanding_moments(self, theta: np.ndarray) -> np.ndarray:
        """Return the scalable block of demanding moments."""

        residual = self.y - self.x @ theta
        return self.z[:, 3:] * residual[:, None]


def run_estimator(fit, data: dict[str, Any]):
    """Construct the moment model and run one estimator."""

    model = LinearMomentModel(data)
    theta_true = np.asarray(data["theta_true"], dtype=float)
    theta0 = np.zeros_like(theta_true)

    return fit(model, theta0)


def main() -> None:
    """Run the demanding-moment scaling experiment."""

    benchmark = GridBenchmark(
        design_factory=DemandingMomentLinearIVDesign,
        grid={
            "demanding_moments": [1, 2, 4, 8],
        },
        base_design_parameters={
            "n": 500,
        },
        estimators={
            "Tractable": partial(run_estimator, fit_tractable),
            "Projection": partial(run_estimator, fit_projection),
            "Full": partial(run_estimator, fit_full),
        },
        repetitions=100,
        seed=123,
    )

    results = benchmark.run()

    comparison = results.compare(
        reference="Full",
        candidate="Projection",
    )

    output_dir = Path("results") / "section5_demanding_moments"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Estimator summaries
    # ------------------------------------------------------------------

    results.to_csv(
        output_dir / "demanding_moment_summary.csv"
    )

    results.to_csv(
        output_dir / "demanding_moment_replications.csv",
        table="replications",
    )

    results.to_latex(
        output_dir / "demanding_moment_summary.tex",
        caption=(
            "Linear IV Monte Carlo results by the number of "
            "demanding moments."
        ),
        label="tab:demanding-moment-scaling",
    )

    results.plot_runtime(
        output_dir / "runtime_by_demanding_moments.pdf",
        x="demanding_moments",
    )

    results.plot_rmse(
        output_dir / "rmse_parameter_0_by_demanding_moments.pdf",
        x="demanding_moments",
        parameter_index=0,
    )

    results.plot_rmse(
        output_dir / "rmse_parameter_1_by_demanding_moments.pdf",
        x="demanding_moments",
        parameter_index=1,
    )

    results.plot_demanding_evaluations(
        output_dir / "evaluations_by_demanding_moments.pdf",
        x="demanding_moments",
    )

    # ------------------------------------------------------------------
    # Paired Projection--Full diagnostics
    # ------------------------------------------------------------------

    comparison.to_csv(
        output_dir / "projection_full_comparison.csv"
    )

    comparison.to_csv(
        output_dir / "projection_full_replications.csv",
        table="replications",
    )

    comparison.to_latex(
        output_dir / "projection_full_comparison.tex",
        caption=(
            "Paired Projection--Full comparison by the number of "
            "demanding moments."
        ),
        label="tab:projection-full-demanding-moments",
    )

    comparison.plot_parameter_distance(
        output_dir / "projection_full_parameter_distance.pdf",
        x="demanding_moments",
    )

    comparison.plot_metric(
        output_dir / "projection_full_covariance_distance.pdf",
        x="demanding_moments",
        metric="mean_covariance_distance",
        ylabel="Mean covariance-matrix distance",
    )

    comparison.plot_runtime_speedup(
        output_dir / "projection_full_runtime_speedup.pdf",
        x="demanding_moments",
    )

    comparison.plot_metric(
        output_dir / "projection_full_evaluation_reduction.pdf",
        x="demanding_moments",
        metric="mean_demanding_evaluation_reduction",
        ylabel="Demanding-evaluation reduction",
    )

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------

    print("\nDemanding-moment scaling benchmark")
    print("=" * 110)
    print(results.summary_table())

    print("\nPaired Projection--Full comparison")
    print("=" * 110)
    print(comparison.summary_table())

    print(f"\nSaved experiment outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

"""Sample-size grid benchmark for the Version 1.0 API."""

from functools import partial
from pathlib import Path

import numpy as np

from orthogmm import (
    GridBenchmark,
    fit_full,
    fit_projection,
    fit_tractable,
)
from orthogmm.simulation import LinearIVDesign


class LinearMomentModel:
    """Linear IV moments split into identified tractable and demanding blocks."""

    def __init__(self, data):
        self.y = data["y"]
        self.x = data["X"]
        self.z = data["Z"]

    def tractable_moments(self, theta):
        residual = self.y - self.x @ theta
        return self.z[:, :3] * residual[:, None]

    def demanding_moments(self, theta):
        residual = self.y - self.x @ theta
        return self.z[:, 3:] * residual[:, None]


def run_estimator(fit, data):
    model = LinearMomentModel(data)
    return fit(model, np.zeros(data["theta_true"].size))


def main() -> None:
    benchmark = GridBenchmark(
        design_factory=LinearIVDesign,
        grid={"n": [250, 500, 1000, 2000]},
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

    output_dir = Path("results") / "linear_iv_grid_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(output_dir / "grid_summary.csv")
    results.to_csv(
        output_dir / "grid_replications.csv",
        table="replications",
    )
    results.to_latex(
        output_dir / "grid_summary.tex",
        caption="Linear IV Monte Carlo benchmark over sample size.",
        label="tab:linear-iv-grid",
    )
    results.plot_runtime(output_dir / "runtime_by_n.pdf", x="n")
    results.plot_rmse(
        output_dir / "rmse_parameter_0_by_n.pdf",
        x="n",
        parameter_index=0,
    )
    results.plot_rmse(
        output_dir / "rmse_parameter_1_by_n.pdf",
        x="n",
        parameter_index=1,
    )
    results.plot_demanding_evaluations(
        output_dir / "demanding_evaluations_by_n.pdf",
        x="n",
    )

    comparison.to_csv(output_dir / "projection_full_grid_comparison.csv")
    comparison.to_csv(
        output_dir / "projection_full_grid_replications.csv",
        table="replications",
    )
    comparison.to_latex(
        output_dir / "projection_full_grid_comparison.tex",
        caption="Paired Projection--Full comparison over sample size.",
        label="tab:projection-full-grid",
    )
    comparison.plot_parameter_distance(
        output_dir / "projection_full_distance_by_n.pdf",
        x="n",
    )
    comparison.plot_runtime_speedup(
        output_dir / "projection_full_speedup_by_n.pdf",
        x="n",
    )

    print("\nLinear IV grid benchmark")
    print("=" * 100)
    print(results.summary_table())
    print("\nPaired Projection--Full comparison by sample size")
    print("=" * 100)
    print(comparison.summary_table())
    print(f"\nSaved grid outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

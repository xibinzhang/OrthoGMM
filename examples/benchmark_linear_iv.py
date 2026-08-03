"""Minimal Version 1.0 benchmark API example."""

from functools import partial
from pathlib import Path

import numpy as np

from orthogmm import (
    MonteCarloBenchmark,
    fit_full,
    fit_projection,
    fit_tractable,
)
from orthogmm.simulation import LinearIVDesign


class LinearMomentModel:
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


benchmark = MonteCarloBenchmark(
    design=LinearIVDesign(n=500),
    estimators={
        "Tractable": partial(run_estimator, fit_tractable),
        "Projection": partial(run_estimator, fit_projection),
        "Full": partial(run_estimator, fit_full),
    },
    repetitions=100,
    seed=123,
)

results = benchmark.run()
output_dir = Path("results") / "linear_iv_benchmark"
output_dir.mkdir(parents=True, exist_ok=True)

results.to_csv(output_dir / "benchmark_summary.csv")
results.to_latex(
    output_dir / "benchmark_summary.tex",
    caption="Linear IV Monte Carlo benchmark.",
    label="tab:linear-iv-benchmark",
)
results.plot_runtime(output_dir / "benchmark_runtime.pdf")
results.plot_rmse(output_dir / "benchmark_rmse.pdf")
results.plot_demanding_evaluations(
    output_dir / "benchmark_demanding_evaluations.pdf"
)

comparison = results.compare(
    reference="Full",
    candidate="Projection",
)
comparison.to_csv(output_dir / "projection_full_comparison.csv")
comparison.to_csv(
    output_dir / "projection_full_replications.csv",
    table="replications",
)
comparison.to_latex(
    output_dir / "projection_full_comparison.tex",
    caption="Paired Projection--Full GMM comparison.",
    label="tab:projection-full-comparison",
)
comparison.plot_parameter_distance(
    output_dir / "projection_full_parameter_distance.pdf"
)
comparison.plot_runtime_speedup(
    output_dir / "projection_full_runtime_speedup.pdf"
)

print("\nLinear IV benchmark summary")
print("=" * 100)
print(results.summary_table())
print("\nPaired Projection--Full comparison")
print("=" * 100)
print(comparison.summary_table())
print(f"\nSaved benchmark outputs to: {output_dir.resolve()}")

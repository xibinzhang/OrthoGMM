"""Integration-cost scaling experiment for OrthoGMM Version 1.0.

The experiment holds the statistical model, sample size, demanding-moment
block, and simulated samples fixed while varying the numerical work performed
inside each demanding-moment evaluation:

    Q = 10, 20, 40, 80.

Common random numbers ensure that statistical results are directly comparable
across Q. Only computation should change systematically.
"""

from __future__ import annotations

from csv import DictWriter
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from orthogmm import GridBenchmark, fit_full, fit_projection, fit_tractable


@dataclass
class IntegrationCostLinearIVDesign:
    """Linear IV design with scalable demanding-evaluation cost."""

    n: int = 500
    demanding_moments: int = 4
    quadrature_nodes: int = 10
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
        if self.quadrature_nodes <= 0:
            raise ValueError("quadrature_nodes must be positive.")
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
            (exogenous_regressor, endogenous_regressor)
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
        y = X @ theta_true + self.sigma * structural_shock

        return {
            "y": y.astype(float),
            "X": X,
            "Z": Z,
            "theta_true": theta_true,
            "n": self.n,
            "demanding_moments": self.demanding_moments,
            "quadrature_nodes": self.quadrature_nodes,
            "seed": effective_seed,
        }


class LinearMomentModel:
    """Linear IV moments with lazily constructed costly evaluations."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.y = np.asarray(data["y"], dtype=float)
        self.x = np.asarray(data["X"], dtype=float)
        self.z = np.asarray(data["Z"], dtype=float)
        self.quadrature_nodes = int(data["quadrature_nodes"])

        if self.y.ndim != 1:
            raise ValueError("data['y'] must be one-dimensional.")
        if self.x.ndim != 2:
            raise ValueError("data['X'] must be two-dimensional.")
        if self.z.ndim != 2:
            raise ValueError("data['Z'] must be two-dimensional.")
        if not (self.y.shape[0] == self.x.shape[0] == self.z.shape[0]):
            raise ValueError(
                "y, X, and Z must contain the same number of observations."
            )
        if self.z.shape[1] < 4:
            raise ValueError(
                "The design must contain at least one demanding moment."
            )

        self._nodes: np.ndarray | None = None
        self._weights: np.ndarray | None = None

    def _quadrature_rule(self) -> tuple[np.ndarray, np.ndarray]:
        """Construct and cache the quadrature rule only when demanded."""

        if self._nodes is None or self._weights is None:
            nodes, weights = np.polynomial.hermite.hermgauss(
                self.quadrature_nodes
            )
            self._nodes = np.asarray(nodes, dtype=float)
            self._weights = np.asarray(weights, dtype=float)
            self._weights /= self._weights.sum()
        return self._nodes, self._weights

    def tractable_moments(self, theta: np.ndarray) -> np.ndarray:
        """Return the identified tractable moment block."""

        residual = self.y - self.x @ theta
        return self.z[:, :3] * residual[:, None]

    def demanding_moments(self, theta: np.ndarray) -> np.ndarray:
        """Return demanding moments after performing quadrature work."""

        residual = self.y - self.x @ theta
        nodes, weights = self._quadrature_rule()

        shifted_residual = (
            residual[:, None] - np.sqrt(2.0) * nodes[None, :]
        )
        quadrature_kernel = np.exp(-0.5 * shifted_residual**2)
        quadrature_integral = quadrature_kernel @ weights

        demanding = self.z[:, 3:] * residual[:, None]
        return demanding + 0.0 * quadrature_integral[:, None]


def run_estimator(fit, data: dict[str, Any]):
    """Construct the moment model and run one estimator."""

    model = LinearMomentModel(data)
    theta_true = np.asarray(data["theta_true"], dtype=float)
    return fit(model, np.zeros_like(theta_true))


def demanding_work_rows(results) -> list[dict[str, Any]]:
    """Create one calls-times-Q work row per cell and estimator."""

    rows = []
    for row in results.summary_rows():
        if row["parameter_index"] != 0:
            continue
        evaluations = row["mean_demanding_evaluations"]
        q = int(row["quadrature_nodes"])
        rows.append(
            {
                "quadrature_nodes": q,
                "estimator": row["estimator"],
                "mean_demanding_evaluations": evaluations,
                "mean_quadrature_work_units": (
                    None if evaluations is None else float(evaluations) * q
                ),
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write flat rows to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_work_units(path: Path, rows: list[dict[str, Any]]) -> None:
    """Plot demanding calls multiplied by quadrature nodes."""

    import matplotlib.pyplot as plt

    estimators = list(dict.fromkeys(row["estimator"] for row in rows))
    figure, axes = plt.subplots()
    for estimator in estimators:
        selected = [row for row in rows if row["estimator"] == estimator]
        selected.sort(key=lambda row: row["quadrature_nodes"])
        axes.plot(
            [row["quadrature_nodes"] for row in selected],
            [row["mean_quadrature_work_units"] for row in selected],
            marker="o",
            label=estimator,
        )
    axes.set_xlabel("Quadrature nodes")
    axes.set_ylabel("Mean quadrature work units (calls × Q)")
    axes.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def main() -> None:
    """Run the common-sample integration-cost scaling experiment."""

    benchmark = GridBenchmark(
        design_factory=IntegrationCostLinearIVDesign,
        grid={"quadrature_nodes": [10, 20, 40, 80]},
        base_design_parameters={"n": 500, "demanding_moments": 4},
        estimators={
            "Tractable": partial(run_estimator, fit_tractable),
            "Projection": partial(run_estimator, fit_projection),
            "Full": partial(run_estimator, fit_full),
        },
        repetitions=100,
        seed=123,
        common_random_numbers=True,
    )

    results = benchmark.run()
    comparison = results.compare(reference="Full", candidate="Projection")

    output_dir = Path("results") / "section5_integration_cost"
    output_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(output_dir / "integration_cost_summary.csv")
    results.to_csv(
        output_dir / "integration_cost_replications.csv",
        table="replications",
    )
    results.to_latex(
        output_dir / "integration_cost_summary.tex",
        caption="Linear IV Monte Carlo results by quadrature complexity.",
        label="tab:integration-cost-scaling",
    )
    results.plot_runtime(
        output_dir / "runtime_by_quadrature_nodes.pdf",
        x="quadrature_nodes",
    )
    results.plot_rmse(
        output_dir / "rmse_parameter_0_by_quadrature_nodes.pdf",
        x="quadrature_nodes",
        parameter_index=0,
    )
    results.plot_rmse(
        output_dir / "rmse_parameter_1_by_quadrature_nodes.pdf",
        x="quadrature_nodes",
        parameter_index=1,
    )
    results.plot_demanding_evaluations(
        output_dir / "evaluations_by_quadrature_nodes.pdf",
        x="quadrature_nodes",
    )

    comparison.to_csv(output_dir / "projection_full_comparison.csv")
    comparison.to_csv(
        output_dir / "projection_full_replications.csv",
        table="replications",
    )
    comparison.to_latex(
        output_dir / "projection_full_comparison.tex",
        caption=(
            "Paired Projection--Full comparison by quadrature complexity."
        ),
        label="tab:projection-full-integration-cost",
    )
    comparison.plot_parameter_distance(
        output_dir / "projection_full_parameter_distance.pdf",
        x="quadrature_nodes",
    )
    comparison.plot_metric(
        output_dir / "projection_full_covariance_distance.pdf",
        x="quadrature_nodes",
        metric="mean_covariance_distance",
        ylabel="Mean covariance-matrix distance",
    )
    comparison.plot_runtime_speedup(
        output_dir / "projection_full_runtime_speedup.pdf",
        x="quadrature_nodes",
    )
    comparison.plot_metric(
        output_dir / "projection_full_evaluation_reduction.pdf",
        x="quadrature_nodes",
        metric="mean_demanding_evaluation_reduction",
        ylabel="Demanding-evaluation reduction",
    )

    work_rows = demanding_work_rows(results)
    write_rows(output_dir / "quadrature_work_units.csv", work_rows)
    plot_work_units(
        output_dir / "quadrature_work_units.pdf",
        work_rows,
    )

    print("\nIntegration-cost scaling benchmark")
    print("=" * 110)
    print(results.summary_table())
    print("\nPaired Projection--Full comparison")
    print("=" * 110)
    print(comparison.summary_table())
    print(f"\nSaved experiment outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

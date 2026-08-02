from pathlib import Path

import numpy as np

from orthogmm import (
    MonteCarloBenchmark,
    fit_full,
    fit_full_gmm,
    fit_projection,
    fit_seip,
    fit_tractable,
    fit_tractable_gmm,
)
from orthogmm.simulation import LinearIVDesign


class LinearMomentModel:
    def __init__(self, data):
        self.y = data["y"]
        self.x = data["X"]
        self.z = data["Z"]

    def tractable_moments(self, theta):
        residual = self.y - self.x @ theta
        return self.z[:, :2] * residual[:, None]

    def demanding_moments(self, theta):
        residual = self.y - self.x @ theta
        return self.z[:, 2:] * residual[:, None]


def test_concise_api_matches_backward_compatible_functions() -> None:
    data = LinearIVDesign(n=100, seed=12).generate(seed=34)
    model = LinearMomentModel(data)
    theta0 = np.zeros(2)

    concise = fit_tractable(model, theta0)
    legacy = fit_tractable_gmm(model, theta0)
    np.testing.assert_allclose(concise.theta, legacy.theta)

    concise = fit_full(model, theta0)
    legacy = fit_full_gmm(model, theta0)
    np.testing.assert_allclose(concise.theta, legacy.theta)

    concise = fit_projection(model, theta0)
    legacy = fit_seip(model, theta0)
    np.testing.assert_allclose(concise.theta, legacy.theta)


def test_benchmark_summary_alias_and_csv_export(tmp_path: Path) -> None:
    design = LinearIVDesign(n=50, seed=123)

    def runner(data):
        return {
            "theta": data["theta_true"].copy(),
            "standard_errors": np.array([0.1, 0.1]),
            "success": True,
        }

    results = MonteCarloBenchmark(
        design=design,
        estimators={"test": runner},
        repetitions=2,
        seed=9,
    ).run()

    assert results.summary() == results.summarize()
    summary_path = results.to_csv(tmp_path / "summary.csv")
    records_path = results.to_csv(
        tmp_path / "replications.csv",
        table="replications",
    )
    assert summary_path.exists()
    assert records_path.exists()

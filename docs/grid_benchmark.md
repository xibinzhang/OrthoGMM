# Grid benchmarks

`GridBenchmark` runs the existing `MonteCarloBenchmark` over a Cartesian grid
of design parameters. It does not alter any estimator; each grid cell is an
ordinary benchmark with its own deterministic seed.

```python
from orthogmm import GridBenchmark
from orthogmm.simulation import LinearIVDesign

benchmark = GridBenchmark(
    design_factory=LinearIVDesign,
    grid={
        "n": [250, 500, 1000, 2000],
        "n_instruments": [5, 9],
    },
    estimators=estimators,
    repetitions=500,
    seed=123,
)

results = benchmark.run()
results.to_csv("grid_summary.csv")
results.to_latex("grid_summary.tex")
results.plot_runtime("runtime_by_n.pdf", x="n")

comparison = results.compare(
    reference="Full",
    candidate="Projection",
)
comparison.to_csv("projection_full_grid.csv")
comparison.plot_parameter_distance("distance_by_n.pdf", x="n")
```

The grid runner preserves the insertion order of grid parameters and values.
Cell seeds are derived deterministically from the master seed, so rerunning the
same grid produces the same simulated samples and estimates.

## Main objects

- `GridBenchmark`: defines and runs the Cartesian grid.
- `GridResults`: aggregates benchmark summaries and replication records.
- `GridEstimatorComparison`: aggregates paired candidate-reference diagnostics
  for every design cell.

`GridResults.by_parameters(...)` retrieves a unique completed design cell.
CSV exports support both `table="summary"` and `table="replications"`.

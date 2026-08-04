# Common random numbers in grid benchmarks

`GridBenchmark` supports controlled comparisons across design cells through
`common_random_numbers=True`.

```python
benchmark = GridBenchmark(
    design_factory=MyDesign,
    grid={"quadrature_nodes": [10, 20, 40, 80]},
    estimators=estimators,
    repetitions=500,
    seed=123,
    common_random_numbers=True,
)
```

When enabled, every grid cell passes the same Monte Carlo master seed to its
underlying `MonteCarloBenchmark`. Consequently, the same deterministic
replication-seed sequence is reused across cells. This is appropriate when a
grid parameter changes numerical cost, approximation fidelity, or another
implementation feature while the simulated data should remain fixed.

The default is `False`. Under the default, the master seed is split into a
separate deterministic seed stream for each cell. This remains appropriate for
general design grids in which the cells represent different data-generating
processes.

Common random numbers guarantee matching replication seeds. They produce
identical simulated samples only when each design's `generate(seed=...)` method
uses the random stream compatibly across the relevant grid cells.

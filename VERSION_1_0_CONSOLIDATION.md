# OrthoGMM 1.0 API Consolidation

## Preferred public API

- `fit_tractable(model, theta0, **kwargs)`
- `fit_projection(model, theta0, **kwargs)`
- `fit_full(model, theta0, **kwargs)`

The legacy functions remain fully available:

- `fit_tractable_gmm`
- `fit_seip`
- `fit_full_gmm`

## Benchmark framework

`MonteCarloBenchmark` is the canonical Version 1.0 entry point and extends the existing `Experiment` engine. It preserves reproducible seed spawning, generic estimator runners, failure handling, statistical summaries, runtime accounting, and demanding-moment evaluation counts.

`ExperimentResults` now supports:

- `summary()` and `summarize()`;
- `to_csv(path, table="summary")`;
- `to_csv(path, table="replications")`.

## Validation

- Package version: `1.0.0.dev0`
- Test result: `104 passed, 2 skipped`
- Existing numerical routines were not changed.

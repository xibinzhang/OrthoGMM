# Changelog

## 0.1.0 — 2026-07-26

- Initial reference implementation of tractable GMM, full GMM, and SEIP.
- IID and cluster covariance conventions.
- Analytical or finite-difference Jacobians.
- Matrix regularization, condition diagnostics, operation counts, and timing.
- Abstract multi-fidelity BLP adapter.
- Linear-IV example and test suite.

## 0.2.0.dev0 - architecture refactor started

- Added a formal `BaseMomentModel` contract for unit-level tractable and demanding moments.
- Added estimator classes: `TractableGMM`, `FullGMM`, and `SOPEstimator`.
- Added a standalone `OrthogonalProjection` operator implementing the Section 3 objects
  \(B\), \(S\), \(R\), \(J\), the projected score, and orthogonality diagnostics.
- Added package layers for models, operators, estimators, and diagnostics while retaining
  the original functional API.
- Added architecture tests and bumped the development version to `0.2.0.dev0`.


## 1.0.0.dev0 — API consolidation and benchmark framework

- Added the preferred functional API: `fit_tractable`, `fit_projection`, and `fit_full`.
- Retained `fit_tractable_gmm`, `fit_seip`, and `fit_full_gmm` without behavioural changes.
- Exposed `MonteCarloBenchmark` as the canonical benchmark entry point while retaining `Experiment`.
- Added `ExperimentResults.summary()` as a concise alias for `summarize()`.
- Added dependency-free CSV export for summary statistics and replication-level records.
- Added a complete linear-IV benchmark example.
- Added API-equivalence, benchmark-export, and backward-compatibility tests.
- Bumped the development version to `1.0.0.dev0`.

# OrthoGMM

`orthogmm` is a reference Python implementation of the computational specification for orthogonal-projection GMM under computational heterogeneity.

It provides three estimators using one common model interface:

- tractable GMM;
- full GMM; and
- Sequential Oracle Projection (SOP), implemented through projected GMM.

The package is intentionally model-neutral. A model supplies unit-level tractable and demanding moment contributions, and optionally analytical Jacobians and reconstruction routines.

## Installation

```bash
python -m pip install -e .
```

## Minimal model contract

```python
class MyModel:
    parameter_names = ("beta_0", "beta_1")

    def tractable_moments(self, theta):
        # return shape (n_units, q_g)
        ...

    def demanding_moments(self, theta):
        # return shape (n_units, q_h)
        ...
```

Optional methods:

```python
tractable_jacobian(theta)   # shape (q_g, p)
demanding_jacobian(theta)   # shape (q_h, p)
reconstruct(theta)          # arbitrary model-specific output
unit_ids()                  # shape (n_units,)
```

## Example

```bash
python examples/linear_iv.py
```

## Main API

The preferred Version 1.0 functional API is:

```python
from orthogmm import fit_tractable, fit_projection, fit_full

initial = fit_tractable(model, theta0)
projected = fit_projection(model, theta0)
full = fit_full(model, theta0)
```

The original names remain available for backward compatibility:

```python
from orthogmm import fit_tractable_gmm, fit_full_gmm, fit_seip
```

The result object stores estimates, covariance matrices, projection objects, Jacobians, condition numbers, orthogonality diagnostics, evaluation counts, warnings, and optional reconstruction output.

## Development architecture (v0.2)

The package is being refactored to mirror the revised paper:

- `orthogmm.model`: model contracts and application-specific operators;
- `orthogmm.operators`: model-independent projection objects;
- `orthogmm.estimators`: tractable, full, and Sequential Oracle Projection estimators;
- `orthogmm.diagnostics`: computational accounting and numerical diagnostics.

The original functional interface remains available. The equivalent class interface is:

```python
from orthogmm import SOPEstimator

result = SOPEstimator().fit(model, theta0)
print(result.summary())
```

The standalone projection layer can also be used directly:

```python
from orthogmm import OrthogonalProjection

projection = OrthogonalProjection().fit(g, h, G, H)
B = projection.coefficient
S = projection.residual_covariance
R = projection.residualized_jacobian
J = projection.information
```


## Monte Carlo benchmarks

```python
from orthogmm import MonteCarloBenchmark

benchmark = MonteCarloBenchmark(
    design=design,
    estimators={
        "Tractable": tractable_runner,
        "Projection": projection_runner,
        "Full": full_runner,
    },
    repetitions=500,
    seed=123,
)

results = benchmark.run()
summary = results.summary()
results.to_csv("benchmark_summary.csv")
results.to_csv("benchmark_replications.csv", table="replications")
```

See `examples/benchmark_linear_iv.py` for a complete example.

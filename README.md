# OrthoGMM

`orthogmm` is a reference Python implementation of the computational specification for orthogonal-projection GMM under computational heterogeneity.

It provides three estimators using one common model interface:

- tractable GMM;
- full GMM; and
- Sequential Efficient Influence Projection (SEIP), also called projected GMM.

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

```python
from orthogmm import fit_tractable_gmm, fit_full_gmm, fit_seip
```

The result object stores estimates, covariance matrices, projection objects, Jacobians, condition numbers, orthogonality diagnostics, evaluation counts, warnings, and optional reconstruction output.

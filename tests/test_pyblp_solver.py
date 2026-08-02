import numpy as np
import pytest

pytest.importorskip("pyblp")

from orthogmm import FidelityConfig
from orthogmm.model.petrin import build_petrin_problem
from orthogmm.solvers import PyBLPSolver


@pytest.fixture(scope="module")
def setup():
    return build_petrin_problem()


def test_solver_validates_parameter_shapes(setup) -> None:
    solver = PyBLPSolver()
    fidelity = FidelityConfig(
        name="test",
        draws=10,
        contraction_tolerance=1e-8,
        max_iterations=100,
    )

    with pytest.raises(ValueError):
        solver.solve(
            setup,
            fidelity=fidelity,
            sigma=np.zeros((2, 2)),
            fixed_parameters=True,
        )

    with pytest.raises(ValueError):
        solver.solve(
            setup,
            fidelity=fidelity,
            pi=np.zeros((2, 2)),
            fixed_parameters=True,
        )


def test_solver_rejects_invalid_weight(setup) -> None:
    solver = PyBLPSolver()
    fidelity = FidelityConfig(
        name="test",
        draws=10,
        contraction_tolerance=1e-8,
        max_iterations=100,
    )

    with pytest.raises(ValueError):
        solver.solve(
            setup,
            fidelity=fidelity,
            fixed_parameters=True,
            weighting_matrix=np.ones((2, 3)),
        )


def test_fidelity_options_cannot_override_solver_keys(setup) -> None:
    solver = PyBLPSolver()
    fidelity = FidelityConfig(
        name="test",
        draws=10,
        contraction_tolerance=1e-8,
        max_iterations=100,
        options={"method": "2s"},
    )

    with pytest.raises(ValueError):
        solver.solve(
            setup,
            fidelity=fidelity,
            fixed_parameters=True,
        )

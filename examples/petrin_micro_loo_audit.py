"""Audit full and leave-one-market-out Petrin micro moments.

This script identifies the exact public PyBLP objects needed to construct
market-level micro pseudo-contributions. It performs:

1. one full-sample fixed-parameter evaluation with micro moments;
2. one leave-one-market-out fixed-parameter evaluation at the same sigma/pi;
3. algebraic checks linking ``moments``, ``micro``, and ``micro_values``.

Run from the repository root:

    py examples\petrin_micro_loo_audit.py

Optionally select the omitted market:

    py examples\petrin_micro_loo_audit.py --market 1981
"""

from __future__ import annotations

import argparse

import numpy as np

from orthogmm import FidelityConfig
from orthogmm.model.petrin import build_petrin_problem
from orthogmm.solvers import PyBLPSolver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", type=int, default=None)
    return parser.parse_args()


def vector(value) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1)


def describe(name: str, value) -> None:
    array = np.asarray(value, dtype=float)
    print(
        f"{name:<28} shape={str(array.shape):<12} "
        f"norm={np.linalg.norm(array):.6e}"
    )


def solve_fixed(setup, sigma, pi):
    fidelity = FidelityConfig(
        name="micro_fixed",
        draws=setup.n_agents,
        contraction_tolerance=1e-10,
        max_iterations=1000,
        seed=0,
    )
    return PyBLPSolver().solve(
        setup,
        fidelity=fidelity,
        sigma=sigma,
        pi=pi,
        include_micro=True,
        fixed_parameters=True,
        method="1s",
    )


def main() -> None:
    args = parse_args()

    full_setup = build_petrin_problem()
    market_id = (
        int(full_setup.market_ids[0])
        if args.market is None
        else args.market
    )

    if market_id not in set(map(int, full_setup.market_ids)):
        raise ValueError(f"Unknown market ID: {market_id}")

    sigma = full_setup.initial_sigma
    pi = full_setup.initial_pi

    print(f"Full evaluation at fixed parameters; omitted market={market_id}")
    full_eval = solve_fixed(full_setup, sigma, pi)
    full = full_eval.results

    reduced_setup = build_petrin_problem(
        exclude_market_ids={market_id},
    )
    reduced_eval = solve_fixed(reduced_setup, sigma, pi)
    reduced = reduced_eval.results

    print("\nFull-sample public arrays")
    print("-------------------------")
    for name in (
        "moments",
        "micro",
        "micro_values",
        "micro_covariances",
        "micro_by_theta_jacobian",
    ):
        if hasattr(full, name):
            describe(name, getattr(full, name))
        else:
            print(f"{name:<28} MISSING")

    full_moments = vector(full.moments)
    reduced_moments = vector(reduced.moments)
    full_micro = vector(full.micro)
    reduced_micro = vector(reduced.micro)
    full_values = vector(full.micro_values)
    reduced_values = vector(reduced.micro_values)

    q_micro = full_micro.size
    if q_micro == 0:
        raise RuntimeError(
            "The full fixed evaluation returned no micro moments."
        )

    print("\nDimensions")
    print("----------")
    print(f"Full total moments:       {full_moments.size}")
    print(f"Reduced total moments:    {reduced_moments.size}")
    print(f"Micro moments:            {q_micro}")
    print(f"Full aggregate moments:   {full_moments.size - q_micro}")
    print(f"Reduced markets:          {reduced_setup.n_markets}")

    full_tail = full_moments[-q_micro:]
    reduced_tail = reduced_moments[-q_micro:]

    print("\nPublic-array identities")
    print("-----------------------")
    print(
        "max |moments tail - micro|: "
        f"{np.max(np.abs(full_tail - full_micro)):.6e}"
    )
    print(
        "max |reduced tail - reduced micro|: "
        f"{np.max(np.abs(reduced_tail - reduced_micro)):.6e}"
    )

    targets = np.asarray(
        [moment.value for moment in full_setup.micro_moments],
        dtype=float,
    )
    print(
        "max |micro - (micro_values-targets)|: "
        f"{np.max(np.abs(full_micro - (full_values - targets))):.6e}"
    )
    print(
        "max |micro - (targets-micro_values)|: "
        f"{np.max(np.abs(full_micro - (targets - full_values))):.6e}"
    )

    T = full_setup.n_markets

    # Delete-one pseudo-values. Their average equals the full-sample micro
    # residual vector by construction:
    #
    #     p_t = T r_full - (T-1) r_{(-t)}.
    #
    pseudo = T * full_micro - (T - 1) * reduced_micro

    print("\nSingle-market jackknife pseudo-contribution")
    print("-------------------------------------------")
    print(f"Market: {market_id}")
    print(f"Shape:  {pseudo.shape}")
    print(f"Norm:   {np.linalg.norm(pseudo):.6e}")
    print("Values:")
    print(np.array2string(pseudo, precision=8, suppress_small=False))

    print("\nTimings")
    print("-------")
    print(f"Full fixed micro:       {full_eval.elapsed_seconds:.3f}s")
    print(f"Leave-one-out micro:    {reduced_eval.elapsed_seconds:.3f}s")


if __name__ == "__main__":
    main()

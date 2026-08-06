"""Section 5 Monte Carlo driver with common random numbers and raw diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from scipy.stats import norm

from orthogmm import (
    RandomCoefficientIntegration,
    RandomCoefficientMomentModel,
    fit_full_gmm,
    fit_seip,
    fit_tractable_gmm,
)
from orthogmm.simulation import NonlinearRandomCoefficientDesign


THETA0 = np.array([0.0, 0.0])
BOUNDS = [(-5.0, 5.0), (-5.0, 5.0)]
MASTER_SEED = 19890604
CONFIDENCE_LEVEL = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=("baseline", "basis", "quadrature"),
        default="baseline",
    )
    parser.add_argument("--replications", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
    )
    return parser.parse_args()


def replication_seeds(repetitions: int) -> list[int]:
    sequence = np.random.SeedSequence(MASTER_SEED)
    return [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in sequence.spawn(repetitions)
    ]


def make_model(data, *, basis_k: int, q_nodes: int):
    return RandomCoefficientMomentModel.from_data(
        data,
        basis_k=basis_k,
        basis_family="fourier",
        integration=RandomCoefficientIntegration(
            mode="quadrature",
            q_nodes=q_nodes,
        ),
    )


def standard_accounting(result) -> dict[str, Any]:
    return {
        "primary_success": "",
        "primary_message": "",
        "retry_attempted": False,
        "retry_success": "",
        "retry_message": "",
        "primary_objective_evaluations": "",
        "primary_demanding_evaluations": "",
        "retry_objective_evaluations": "",
        "retry_demanding_evaluations": "",
        "total_objective_evaluations": (
            result.counts.tractable_objective
        ),
        "total_demanding_evaluations": (
            result.counts.demanding_moments_total
        ),
    }


def fit_full_with_recovery(model):
    primary = fit_full_gmm(
        model,
        theta0=THETA0,
        bounds=BOUNDS,
    )

    accounting = {
        "primary_success": bool(primary.success),
        "primary_message": str(primary.message),
        "retry_attempted": False,
        "retry_success": "",
        "retry_message": "",
        "primary_objective_evaluations": (
            primary.counts.tractable_objective
        ),
        "primary_demanding_evaluations": (
            primary.counts.demanding_moments_total
        ),
        "retry_objective_evaluations": 0,
        "retry_demanding_evaluations": 0,
        "total_objective_evaluations": (
            primary.counts.tractable_objective
        ),
        "total_demanding_evaluations": (
            primary.counts.demanding_moments_total
        ),
    }

    if primary.success:
        return primary, accounting

    accounting["retry_attempted"] = True

    retry = fit_full_gmm(
        model,
        theta0=np.asarray(primary.theta, dtype=float),
        bounds=BOUNDS,
        optimizer_method="L-BFGS-B",
        optimizer_options={
            "maxiter": 5000,
            "maxls": 100,
            "ftol": 1e-12,
            "gtol": 1e-8,
        },
    )

    accounting["retry_success"] = bool(retry.success)
    accounting["retry_message"] = str(retry.message)
    accounting["retry_objective_evaluations"] = (
        retry.counts.tractable_objective
    )
    accounting["retry_demanding_evaluations"] = (
        retry.counts.demanding_moments_total
    )
    accounting["total_objective_evaluations"] += (
        retry.counts.tractable_objective
    )
    accounting["total_demanding_evaluations"] += (
        retry.counts.demanding_moments_total
    )

    return retry, accounting


def fit_one(estimator, data, *, basis_k: int, q_nodes: int):
    model = make_model(
        data,
        basis_k=basis_k,
        q_nodes=q_nodes,
    )
    if estimator == "Initial":
        result = fit_tractable_gmm(
            model,
            theta0=THETA0,
            bounds=BOUNDS,
        )
        return result, standard_accounting(result)
    if estimator == "Full":
        return fit_full_with_recovery(model)

    result = fit_seip(
        model,
        theta0=THETA0,
        bounds=BOUNDS,
        ridge=1e-8,
    )
    return result, standard_accounting(result)


def optional_float(value: Any) -> float | str:
    if value is None:
        return ""
    scalar = float(value)
    return scalar if np.isfinite(scalar) else ""


def optional_norm(value: Any) -> float | str:
    if value is None:
        return ""
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        return ""
    return float(np.linalg.norm(array))


def optional_component(
    value: Any,
    index: int,
) -> float | str:
    if value is None:
        return ""
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or index >= array.size:
        return ""
    scalar = float(array[index])
    return scalar if np.isfinite(scalar) else ""


def optimizer_fields(result) -> dict[str, Any]:
    out = {
        "stage_one_success": "",
        "stage_one_message": "",
        "stage_one_nfev": "",
        "stage_two_success": "",
        "stage_two_message": "",
        "stage_two_nfev": "",
    }
    value = result.optimizer_result
    if isinstance(value, dict):
        for stage_name in ("stage_one", "stage_two"):
            stage = value.get(stage_name)
            if stage is not None:
                out[f"{stage_name}_success"] = bool(stage.success)
                out[f"{stage_name}_message"] = str(stage.message)
                out[f"{stage_name}_nfev"] = int(stage.nfev)
    elif value is not None:
        out["stage_one_success"] = bool(value.success)
        out["stage_one_message"] = str(value.message)
        out["stage_one_nfev"] = int(value.nfev)
    return out


def result_row(
    *,
    replication: int,
    seed: int,
    estimator: str,
    n: int,
    basis_k: int,
    q_nodes: int,
    truth: np.ndarray,
    result,
    elapsed: float,
    recovery: dict[str, Any] | None = None,
) -> dict[str, Any]:
    recovery = recovery or {}
    row = {
        "replication": replication,
        "replication_seed": seed,
        "estimator": estimator,
        "n": n,
        "basis_k": basis_k,
        "q_nodes": q_nodes,
        "true_alpha": float(truth[0]),
        "true_beta": float(truth[1]),
        "alpha": float(result.theta[0]),
        "beta": float(result.theta[1]),
        "se_alpha": float(result.standard_errors[0]),
        "se_beta": float(result.standard_errors[1]),
        "success": bool(result.success),
        "message": str(result.message),
        "error": "",
        "wall_time_seconds": elapsed,
        "objective_evaluations": recovery.get(
            "total_objective_evaluations",
            result.counts.tractable_objective,
        ),
        "demanding_evaluations": recovery.get(
            "total_demanding_evaluations",
            result.counts.demanding_moments_total,
        ),
        "initial_tractable_alpha": optional_component(
            getattr(result, "initial_tractable_theta", None),
            0,
        ),
        "initial_tractable_beta": optional_component(
            getattr(result, "initial_tractable_theta", None),
            1,
        ),
        "preliminary_alpha": optional_component(
            getattr(result, "preliminary_theta", None),
            0,
        ),
        "preliminary_beta": optional_component(
            getattr(result, "preliminary_theta", None),
            1,
        ),
        "tractable_foc_norm": optional_float(
            getattr(result, "tractable_foc_norm", None)
        ),
        "update_difference_norm": optional_float(
            getattr(result, "update_difference_norm", None)
        ),
        "raw_update_norm": optional_norm(
            getattr(result, "raw_update", None)
        ),
        "full_score_update_norm": optional_norm(
            getattr(result, "full_score_update", None)
        ),
        "residual_only_update_norm": optional_norm(
            getattr(result, "residual_only_update", None)
        ),
        "damping_factor": optional_float(
            getattr(result, "damping_factor", None)
        ),
        "primary_success": recovery.get(
            "primary_success",
            "",
        ),
        "primary_message": recovery.get(
            "primary_message",
            "",
        ),
        "retry_attempted": recovery.get(
            "retry_attempted",
            False,
        ),
        "retry_success": recovery.get(
            "retry_success",
            "",
        ),
        "retry_message": recovery.get(
            "retry_message",
            "",
        ),
        "primary_objective_evaluations": recovery.get(
            "primary_objective_evaluations",
            "",
        ),
        "primary_demanding_evaluations": recovery.get(
            "primary_demanding_evaluations",
            "",
        ),
        "retry_objective_evaluations": recovery.get(
            "retry_objective_evaluations",
            "",
        ),
        "retry_demanding_evaluations": recovery.get(
            "retry_demanding_evaluations",
            "",
        ),
        "warnings": " | ".join(result.warnings),
        "condition_numbers": json.dumps(
            result.condition_numbers,
            sort_keys=True,
        ),
        "effective_ranks": json.dumps(
            result.effective_ranks,
            sort_keys=True,
        ),
        "regularization_method": result.regularization.method,
        "regularization_omega_gg": (
            result.regularization.omega_gg
        ),
        "regularization_residual_covariance": (
            result.regularization.residual_covariance
        ),
        "regularization_information": (
            result.regularization.information
        ),
    }
    row.update(optimizer_fields(result))
    return row


def failure_row(
    *,
    replication: int,
    seed: int,
    estimator: str,
    n: int,
    basis_k: int,
    q_nodes: int,
    truth: np.ndarray,
    error: Exception,
    elapsed: float,
) -> dict[str, Any]:
    row = {
        "replication": replication,
        "replication_seed": seed,
        "estimator": estimator,
        "n": n,
        "basis_k": basis_k,
        "q_nodes": q_nodes,
        "true_alpha": float(truth[0]),
        "true_beta": float(truth[1]),
        "alpha": "",
        "beta": "",
        "se_alpha": "",
        "se_beta": "",
        "success": False,
        "message": "",
        "error": f"{type(error).__name__}: {error}",
        "wall_time_seconds": elapsed,
        "objective_evaluations": "",
        "demanding_evaluations": "",
        "initial_tractable_alpha": "",
        "initial_tractable_beta": "",
        "preliminary_alpha": "",
        "preliminary_beta": "",
        "tractable_foc_norm": "",
        "update_difference_norm": "",
        "raw_update_norm": "",
        "full_score_update_norm": "",
        "residual_only_update_norm": "",
        "damping_factor": "",
        "primary_success": "",
        "primary_message": "",
        "retry_attempted": "",
        "retry_success": "",
        "retry_message": "",
        "primary_objective_evaluations": "",
        "primary_demanding_evaluations": "",
        "retry_objective_evaluations": "",
        "retry_demanding_evaluations": "",
        "warnings": "",
        "condition_numbers": "",
        "effective_ranks": "",
        "regularization_method": "",
        "regularization_omega_gg": "",
        "regularization_residual_covariance": "",
        "regularization_information": "",
    }
    row.update(
        {
            "stage_one_success": "",
            "stage_one_message": "",
            "stage_one_nfev": "",
            "stage_two_success": "",
            "stage_two_message": "",
            "stage_two_nfev": "",
        }
    )
    return row


def run_cell(
    *,
    n: int,
    basis_k: int,
    q_nodes: int,
    seeds: list[int],
) -> list[dict[str, Any]]:
    design = NonlinearRandomCoefficientDesign(
        n=n,
        alpha=0.0,
        beta=1.0,
        lambda_=0.5,
        sigma=0.8,
    )
    rows = []
    for replication, seed in enumerate(seeds, start=1):
        data = design.generate(seed=seed)
        truth = np.asarray(data["theta_true"], dtype=float)
        for estimator in ("Initial", "Full", "SOP"):
            start = perf_counter()
            try:
                result, recovery = fit_one(
                    estimator,
                    data,
                    basis_k=basis_k,
                    q_nodes=q_nodes,
                )
                rows.append(
                    result_row(
                        replication=replication,
                        seed=seed,
                        estimator=estimator,
                        n=n,
                        basis_k=basis_k,
                        q_nodes=q_nodes,
                        truth=truth,
                        result=result,
                        elapsed=perf_counter() - start,
                        recovery=recovery,
                    )
                )
            except Exception as error:
                rows.append(
                    failure_row(
                        replication=replication,
                        seed=seed,
                        estimator=estimator,
                        n=n,
                        basis_k=basis_k,
                        q_nodes=q_nodes,
                        truth=truth,
                        error=error,
                        elapsed=perf_counter() - start,
                    )
                )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    critical = float(
        norm.ppf(0.5 + CONFIDENCE_LEVEL / 2.0)
    )
    out = []
    cells = sorted(
        {
            (
                int(row["n"]),
                int(row["basis_k"]),
                int(row["q_nodes"]),
            )
            for row in rows
        }
    )

    for n, basis_k, q_nodes in cells:
        cell = [
            row for row in rows
            if (
                int(row["n"]) == n
                and int(row["basis_k"]) == basis_k
                and int(row["q_nodes"]) == q_nodes
            )
        ]
        for estimator in ("Initial", "Full", "SOP"):
            est_rows = [
                row for row in cell
                if row["estimator"] == estimator
            ]
            successful = [
                row for row in est_rows
                if bool(row["success"])
            ]
            for index, parameter in enumerate(
                ("alpha", "beta")
            ):
                truth = float(
                    est_rows[0][f"true_{parameter}"]
                )
                estimates = np.asarray(
                    [float(row[parameter]) for row in successful]
                )
                ses = np.asarray(
                    [
                        float(row[f"se_{parameter}"])
                        for row in successful
                    ]
                )

                if estimates.size:
                    errors = estimates - truth
                    bias = float(np.mean(errors))
                    rmse = float(
                        np.sqrt(np.mean(errors**2))
                    )
                    empirical_sd = float(
                        np.std(estimates, ddof=1)
                    )
                    mean_se = float(np.mean(ses))
                    coverage = float(
                        np.mean(
                            np.abs(errors) <= critical * ses
                        )
                    )
                    mean_runtime = float(
                        np.mean(
                            [
                                float(
                                    row["wall_time_seconds"]
                                )
                                for row in successful
                            ]
                        )
                    )
                    mean_objective = float(
                        np.mean(
                            [
                                float(
                                    row[
                                        "objective_evaluations"
                                    ]
                                )
                                for row in successful
                            ]
                        )
                    )
                    mean_demanding = float(
                        np.mean(
                            [
                                float(
                                    row[
                                        "demanding_evaluations"
                                    ]
                                )
                                for row in successful
                            ]
                        )
                    )
                else:
                    bias = rmse = empirical_sd = float("nan")
                    mean_se = coverage = float("nan")
                    mean_runtime = float("nan")
                    mean_objective = float("nan")
                    mean_demanding = float("nan")

                out.append(
                    {
                        "estimator": estimator,
                        "parameter_index": index,
                        "true_value": truth,
                        "repetitions": len(est_rows),
                        "successes": len(successful),
                        "success_rate": (
                            len(successful) / len(est_rows)
                        ),
                        "bias": bias,
                        "rmse": rmse,
                        "empirical_sd": empirical_sd,
                        "mean_standard_error": mean_se,
                        "coverage": coverage,
                        "mean_runtime_seconds": mean_runtime,
                        "mean_objective_evaluations": (
                            mean_objective
                        ),
                        "mean_demanding_evaluations": (
                            mean_demanding
                        ),
                        "n": n,
                        "basis_k": basis_k,
                        "q_nodes": q_nodes,
                        "failed_runs": (
                            len(est_rows) - len(successful)
                        ),
                    }
                )
    return out


def save(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def cells(experiment: str):
    if experiment == "baseline":
        return [
            (n, 1, 40)
            for n in (250, 500, 1000, 2000)
        ]
    if experiment == "basis":
        return [
            (1000, k, 40)
            for k in (1, 2, 4, 8)
        ]
    return [
        (1000, 1, q)
        for q in (10, 20, 40, 80)
    ]


def main() -> None:
    args = parse_args()
    if args.replications <= 0:
        raise ValueError("replications must be positive.")

    shared_seeds = replication_seeds(args.replications)
    raw_rows = []

    for n, basis_k, q_nodes in cells(args.experiment):
        print(
            f"{args.experiment}: "
            f"n={n}, K={basis_k}, Q={q_nodes}, "
            f"R={args.replications}"
        )
        raw_rows.extend(
            run_cell(
                n=n,
                basis_k=basis_k,
                q_nodes=q_nodes,
                seeds=shared_seeds,
            )
        )

    summary_rows = summarize(raw_rows)

    raw_path = (
        args.output_dir
        / f"section5_{args.experiment}_R{args.replications}_raw.csv"
    )
    summary_path = (
        args.output_dir
        / f"section5_{args.experiment}_R{args.replications}.csv"
    )

    save(raw_rows, raw_path)
    save(summary_rows, summary_path)

    print(f"\nSaved raw results to {raw_path}")
    print(f"Saved summary results to {summary_path}")
    print(
        "Failed estimator runs: "
        f"{sum(not bool(row['success']) for row in raw_rows)}"
    )


if __name__ == "__main__":
    main()

"""PyBLP Petrin automobile benchmark builder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class PetrinProblem:
    problem: Any
    micro_moments: tuple[Any, ...]
    initial_sigma: np.ndarray
    initial_pi: np.ndarray
    product_data: pd.DataFrame
    agent_data: pd.DataFrame
    market_ids: np.ndarray

    @property
    def n_markets(self) -> int:
        return int(self.market_ids.size)

    @property
    def n_products(self) -> int:
        return int(self.product_data.shape[0])

    @property
    def n_agents(self) -> int:
        return int(self.agent_data.shape[0])


def build_petrin_problem(
    *,
    exclude_market_ids: Iterable[Any] | None = None,
) -> PetrinProblem:
    """Construct the canonical PyBLP Petrin benchmark without estimation."""

    try:
        import pyblp
    except ImportError as exc:
        raise ImportError(
            "PyBLP is required to build the Petrin benchmark."
        ) from exc

    excluded = set() if exclude_market_ids is None else set(exclude_market_ids)

    product_data = pd.read_csv(pyblp.data.PETRIN_PRODUCTS_LOCATION)
    agent_data = pd.read_csv(pyblp.data.PETRIN_AGENTS_LOCATION)

    if excluded:
        product_data = product_data.loc[
            ~product_data["market_ids"].isin(excluded)
        ].copy()
        agent_data = agent_data.loc[
            ~agent_data["market_ids"].isin(excluded)
        ].copy()

    if product_data.empty:
        raise ValueError("No product observations remain after market exclusion.")
    if agent_data.empty:
        raise ValueError("No agent observations remain after market exclusion.")

    market_ids = pd.unique(product_data["market_ids"])

    product_formulations = (
        pyblp.Formulation(
            "1 + hpwt + space + air + mpd + fwd + mi + sw + su + pv + "
            "pgnp + trend + trend2"
        ),
        pyblp.Formulation(
            "1 + I(-prices) + hpwt + space + air + mpd + fwd + mi + sw + su + pv"
        ),
        pyblp.Formulation(
            "1 + log(hpwt) + log(wt) + log(mpg) + air + fwd + "
            "trend * (jp + eu) + log(q)"
        ),
    )

    agent_formulation = pyblp.Formulation(
        "1 + I(low / income) + I(mid / income) + I(high / income) + "
        "I(log(fs) * fv) + age + fs + mid + high"
    )

    problem = pyblp.Problem(
        product_formulations,
        product_data,
        agent_formulation,
        agent_data,
        costs_type="log",
    )

    micro_dataset = pyblp.MicroDataset(
        name="CEX",
        observations=29125,
        compute_weights=lambda t, p, a: np.ones((a.size, 1 + p.size)),
    )

    def product_indicator(index: int):
        return lambda t, p, a: np.outer(
            np.ones(a.size), np.r_[0, p.X2[:, index]]
        )

    def demographic_product(demographic_index: int, product_index: int):
        return lambda t, p, a: np.outer(
            a.demographics[:, demographic_index],
            np.r_[0, p.X2[:, product_index]],
        )

    parts: dict[str, Any] = {}
    product_indices = {"mi": 7, "sw": 8, "su": 9, "pv": 10}

    for product, index in product_indices.items():
        parts[product] = pyblp.MicroPart(
            name=f"E[{product}_j]",
            dataset=micro_dataset,
            compute_values=product_indicator(index),
        )
        parts[f"age_{product}"] = pyblp.MicroPart(
            name=f"E[age_i * {product}_j]",
            dataset=micro_dataset,
            compute_values=demographic_product(5, index),
        )
        parts[f"fs_{product}"] = pyblp.MicroPart(
            name=f"E[fs_i * {product}_j]",
            dataset=micro_dataset,
            compute_values=demographic_product(6, index),
        )

    inside_mid = pyblp.MicroPart(
        name="E[1{j > 0} * mid_i]",
        dataset=micro_dataset,
        compute_values=lambda t, p, a: np.outer(
            a.demographics[:, 7], np.r_[0, p.X2[:, 0]]
        ),
    )
    inside_high = pyblp.MicroPart(
        name="E[1{j > 0} * high_i]",
        dataset=micro_dataset,
        compute_values=lambda t, p, a: np.outer(
            a.demographics[:, 8], np.r_[0, p.X2[:, 0]]
        ),
    )
    mid = pyblp.MicroPart(
        name="E[mid_i]",
        dataset=micro_dataset,
        compute_values=lambda t, p, a: np.outer(
            a.demographics[:, 7], np.r_[1, p.X2[:, 0]]
        ),
    )
    high = pyblp.MicroPart(
        name="E[high_i]",
        dataset=micro_dataset,
        compute_values=lambda t, p, a: np.outer(
            a.demographics[:, 8], np.r_[1, p.X2[:, 0]]
        ),
    )

    compute_ratio = lambda v: v[0] / v[1]
    compute_ratio_gradient = lambda v: [1 / v[1], -v[0] / v[1] ** 2]

    targets = {
        "age_mi": 0.783,
        "age_sw": 0.730,
        "age_su": 0.740,
        "age_pv": 0.652,
        "fs_mi": 3.86,
        "fs_sw": 3.17,
        "fs_su": 2.97,
        "fs_pv": 3.47,
    }

    micro_moments: list[Any] = []
    for name, value in targets.items():
        demographic, product = name.split("_", maxsplit=1)
        micro_moments.append(
            pyblp.MicroMoment(
                name=f"E[{demographic}_i | {product}_j]",
                value=value,
                parts=[parts[name], parts[product]],
                compute_value=compute_ratio,
                compute_gradient=compute_ratio_gradient,
            )
        )

    micro_moments.extend(
        [
            pyblp.MicroMoment(
                name="E[1{j > 0} | mid_i]",
                value=0.0794,
                parts=[inside_mid, mid],
                compute_value=compute_ratio,
                compute_gradient=compute_ratio_gradient,
            ),
            pyblp.MicroMoment(
                name="E[1{j > 0} | high_i]",
                value=0.1581,
                parts=[inside_high, high],
                compute_value=compute_ratio,
                compute_gradient=compute_ratio_gradient,
            ),
        ]
    )

    initial_sigma = np.diag(
        [3.23, 0, 4.43, 0.46, 0.01, 2.58, 4.42, 0, 0, 0, 0]
    )

    initial_pi = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 7.52, 31.13, 34.49, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.57, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.28, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.31, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.42, 0, 0, 0, 0],
        ],
        dtype=float,
    )

    return PetrinProblem(
        problem=problem,
        micro_moments=tuple(micro_moments),
        initial_sigma=initial_sigma,
        initial_pi=initial_pi,
        product_data=product_data,
        agent_data=agent_data,
        market_ids=np.asarray(market_ids),
    )

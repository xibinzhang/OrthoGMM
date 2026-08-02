"""Market-level aggregate IV moment construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AggregateIVMoments:
    """Market-level demand and supply IV moment contributions."""

    market_ids: NDArray
    demand: FloatArray
    supply: FloatArray

    @property
    def combined(self) -> FloatArray:
        return np.column_stack((self.demand, self.supply))

    @property
    def average(self) -> FloatArray:
        return self.combined.mean(axis=0)


class AggregateIVMomentBuilder:
    """Build market contributions that reproduce PyBLP aggregate IV moments.

    PyBLP's aggregate demand and supply moments have the form

    .. math::

        \\bar g_D = N^{-1} \\sum_j z_{Dj}\\xi_j,
        \\qquad
        \\bar g_S = N^{-1} \\sum_j z_{Sj}\\omega_j.

    With markets as statistical units, this builder defines

    .. math::

        g_{Dt} = (T/N) \\sum_{j \\in t} z_{Dj}\\xi_j,
        \\qquad
        g_{St} = (T/N) \\sum_{j \\in t} z_{Sj}\\omega_j,

    so that the average across markets exactly equals PyBLP's aggregate
    product-level average.
    """

    def build(
        self,
        *,
        product_market_ids: NDArray,
        demand_instruments: FloatArray,
        demand_residuals: FloatArray,
        supply_instruments: FloatArray | None = None,
        supply_residuals: FloatArray | None = None,
        market_order: NDArray | None = None,
    ) -> AggregateIVMoments:
        market_ids = np.asarray(product_market_ids)
        ZD = self._matrix(demand_instruments, "demand_instruments")
        xi = self._vector(demand_residuals, "demand_residuals")

        if market_ids.ndim != 1:
            raise ValueError(
                "product_market_ids must be one-dimensional."
            )

        n_products = market_ids.size

        if ZD.shape[0] != n_products or xi.size != n_products:
            raise ValueError(
                "Demand instruments, residuals, and market identifiers "
                "must contain the same number of products."
            )

        if (supply_instruments is None) != (supply_residuals is None):
            raise ValueError(
                "supply_instruments and supply_residuals must be "
                "supplied together."
            )

        if supply_instruments is None:
            ZS = np.empty((n_products, 0), dtype=float)
            omega = np.empty(n_products, dtype=float)
        else:
            ZS = self._matrix(
                supply_instruments,
                "supply_instruments",
            )
            omega = self._vector(
                supply_residuals,
                "supply_residuals",
            )
            if ZS.shape[0] != n_products or omega.size != n_products:
                raise ValueError(
                    "Supply instruments, residuals, and market "
                    "identifiers must contain the same number of "
                    "products."
                )

        if market_order is None:
            units = np.unique(market_ids)
        else:
            units = np.asarray(market_order)
            if units.ndim != 1:
                raise ValueError(
                    "market_order must be one-dimensional."
                )
            if np.unique(units).size != units.size:
                raise ValueError(
                    "market_order must contain unique identifiers."
                )
            observed = set(np.unique(market_ids).tolist())
            requested = set(units.tolist())
            if observed != requested:
                raise ValueError(
                    "market_order must contain exactly the observed "
                    "market identifiers."
                )

        n_markets = units.size
        scale = n_markets / n_products

        demand = np.empty((n_markets, ZD.shape[1]), dtype=float)
        supply = np.empty((n_markets, ZS.shape[1]), dtype=float)

        for row, market_id in enumerate(units):
            mask = market_ids == market_id
            demand[row] = scale * (ZD[mask].T @ xi[mask])
            if ZS.shape[1]:
                supply[row] = scale * (ZS[mask].T @ omega[mask])

        return AggregateIVMoments(
            market_ids=units,
            demand=demand,
            supply=supply,
        )

    def from_pyblp(
        self,
        problem: Any,
        results: Any,
    ) -> AggregateIVMoments:
        """Build moments from public PyBLP problem and result objects.

        The public ``Problem.products`` object stores the already-constructed
        full demand and supply instrument matrices as ``ZD`` and ``ZS``.
        ``ProblemResults`` stores the corresponding residual vectors as
        ``xi`` and ``omega``.
        """

        if not hasattr(problem, "products"):
            raise AttributeError(
                "PyBLP problem does not expose 'products'."
            )

        products = problem.products

        required_product_fields = ("market_ids", "ZD", "ZS")
        for name in required_product_fields:
            if not hasattr(products, name):
                raise AttributeError(
                    f"PyBLP problem.products does not expose {name!r}."
                )

        if not hasattr(results, "xi"):
            raise AttributeError(
                "PyBLP results do not expose 'xi'."
            )
        if not hasattr(results, "omega"):
            raise AttributeError(
                "PyBLP results do not expose 'omega'."
            )

        market_order = (
            np.asarray(problem.unique_market_ids)
            if hasattr(problem, "unique_market_ids")
            else None
        )

        return self.build(
            product_market_ids=np.asarray(
                products.market_ids
            ).reshape(-1),
            demand_instruments=np.asarray(
                products.ZD,
                dtype=float,
            ),
            demand_residuals=np.asarray(
                results.xi,
                dtype=float,
            ).reshape(-1),
            supply_instruments=np.asarray(
                products.ZS,
                dtype=float,
            ),
            supply_residuals=np.asarray(
                results.omega,
                dtype=float,
            ).reshape(-1),
            market_order=market_order,
        )

    @staticmethod
    def _vector(value: Any, name: str) -> FloatArray:
        array = np.asarray(value, dtype=float).reshape(-1)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values.")
        return array

    @staticmethod
    def _matrix(value: Any, name: str) -> FloatArray:
        array = np.asarray(value, dtype=float)
        if array.ndim != 2:
            raise ValueError(f"{name} must be two-dimensional.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values.")
        return array

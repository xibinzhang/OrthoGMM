from __future__ import annotations

import numpy as np

from .exceptions import ModelContractError
from .types import Array


def _as_2d(x: Array, name: str) -> Array:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ModelContractError(f"{name} must be a two-dimensional unit-by-moment array.")
    if x.shape[0] < 2:
        raise ModelContractError(f"{name} must contain at least two statistical units.")
    if not np.all(np.isfinite(x)):
        raise ModelContractError(f"{name} contains non-finite values.")
    return x


def center(x: Array) -> tuple[Array, Array]:
    x = _as_2d(x, "moments")
    mean = x.mean(axis=0)
    return x - mean, mean


def covariance_blocks_iid(g: Array, h: Array) -> tuple[Array, Array, Array]:
    g = _as_2d(g, "g")
    h = _as_2d(h, "h")
    if g.shape[0] != h.shape[0]:
        raise ModelContractError("Tractable and demanding moments must use the same units.")
    gc, _ = center(g)
    hc, _ = center(h)
    n = g.shape[0]
    return gc.T @ gc / n, hc.T @ gc / n, hc.T @ hc / n


def _cluster_sums(x: Array, clusters: Array) -> Array:
    x = _as_2d(x, "moments")
    clusters = np.asarray(clusters)
    if clusters.ndim != 1 or clusters.shape[0] != x.shape[0]:
        raise ModelContractError("clusters must be one-dimensional with one id per unit.")
    unique, inverse = np.unique(clusters, return_inverse=True)
    sums = np.zeros((len(unique), x.shape[1]), dtype=float)
    np.add.at(sums, inverse, x)
    return sums


def covariance_blocks_cluster(
    g: Array, h: Array, clusters: Array
) -> tuple[Array, Array, Array]:
    g = _as_2d(g, "g")
    h = _as_2d(h, "h")
    if g.shape[0] != h.shape[0]:
        raise ModelContractError("Tractable and demanding moments must use the same units.")
    gs = _cluster_sums(g, clusters)
    hs = _cluster_sums(h, clusters)
    if gs.shape[0] < 2:
        raise ModelContractError("Cluster covariance requires at least two clusters.")
    gsc, _ = center(gs)
    hsc, _ = center(hs)
    # Scale by number of original units, matching covariance of unit averages.
    n = g.shape[0]
    return gsc.T @ gsc / n, hsc.T @ gsc / n, hsc.T @ hsc / n


def residual_covariance_iid(nu: Array) -> Array:
    nuc, _ = center(nu)
    return nuc.T @ nuc / nu.shape[0]


def residual_covariance_cluster(nu: Array, clusters: Array) -> Array:
    nus = _cluster_sums(nu, clusters)
    nusc, _ = center(nus)
    return nusc.T @ nusc / nu.shape[0]

"""
Intrinsic dimensionality estimation for the Structure Discovery phase.
Combines three independent estimators — eigenvalue gap, local PCA, and the
Levina-Bickel MLE — and returns their median as the consensus dimension.
"""
from __future__ import annotations

import numpy as np

from common.logging import get_logger
from common.nn_utils import build_faiss_index, query_knn

_log = get_logger(__name__)

# Neighbourhood size for local PCA and Levina-Bickel MLE.
_K_LOCAL = 20
# Fraction of variance that local PCA must explain to determine dimension.
_VARIANCE_THRESHOLD = 0.95
# Minimum k needed to compute Levina-Bickel (k-1 actual neighbours required).
_MIN_K_MLE = 3


def estimate(data: np.ndarray) -> int:
    """Estimate the intrinsic dimensionality of *data* by consensus of three methods.

    Parameters
    ----------
    data : (n, d) float64 array.

    Returns
    -------
    int : consensus intrinsic dimension in [1, d-1].
    """
    _n, d = data.shape
    gap = _eigenvalue_gap(data)
    _log.info("Eigenvalue-gap estimate: %d", gap)
    local = _local_pca_estimate(data, k=_K_LOCAL)
    _log.info("Local-PCA estimate: %d", local)
    mle = _mle_estimate(data)
    _log.info("MLE (Levina-Bickel) estimate: %d", mle)
    return _consensus_dimension([gap, local, mle], n_features=d)


def _eigenvalue_gap(data: np.ndarray) -> int:
    """Estimate dimension from the largest ratio of consecutive PCA eigenvalues.

    The position of the maximum ratio indicates where significant variance ends.
    """
    cov = np.cov(data.T)
    # eigvalsh returns ascending order; reverse for descending (largest first).
    eigenvalues = np.linalg.eigvalsh(cov)[::-1]
    positive = eigenvalues[eigenvalues > 1e-10]
    if len(positive) < 2:
        return 1
    # Ratio spike at index i means eigenvalues[i] >> eigenvalues[i+1].
    ratios = positive[:-1] / positive[1:]
    return max(1, int(np.argmax(ratios)) + 1)


def _local_pca_estimate(data: np.ndarray, k: int = _K_LOCAL) -> int:
    """Average local PCA dimension across a sample of neighbourhood patches.

    Falls back to the global eigenvalue-gap estimate when n is too small for
    local analysis (n ≤ k).
    """
    n = data.shape[0]
    if n <= k:
        # Not enough points for local analysis; use global estimate.
        return _eigenvalue_gap(data)
    index = build_faiss_index(data)
    # Sample up to 50 seed points for efficiency on large datasets.
    n_seeds = min(50, n)
    seeds = np.linspace(0, n - 1, n_seeds, dtype=int)
    local_dims = []
    for i in seeds:
        # Request k neighbours; index contains the seed, so position 0 is self.
        _d, idx = query_knn(index, data[i : i + 1], k=min(k, n - 1))
        neighbourhood = data[idx[0]]
        local_dims.append(_pca_dim(neighbourhood))
    return max(1, int(np.median(local_dims)))


def _pca_dim(points: np.ndarray) -> int:
    """Return the PCA dimension explaining _VARIANCE_THRESHOLD of variance."""
    if points.shape[0] < 2:
        # Single-point neighbourhood; dimension is undefined — return full dim.
        return points.shape[1]
    cov = np.cov(points.T)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
    total = eigenvalues.sum()
    if total < 1e-10:
        return 1
    cumvar = np.cumsum(eigenvalues) / total
    # searchsorted finds the first index where cumvar >= threshold.
    return max(1, int(np.searchsorted(cumvar, _VARIANCE_THRESHOLD)) + 1)


def _mle_estimate(data: np.ndarray) -> int:
    """Levina-Bickel MLE intrinsic dimension estimator.

    Uses the distribution of pairwise distances in local neighbourhoods to
    estimate the local intrinsic dimension at each point.
    """
    n = data.shape[0]
    # Need at least _MIN_K_MLE neighbours (including self at position 0).
    k = min(_K_LOCAL, n - 1)
    if k < _MIN_K_MLE:
        return 1
    index = build_faiss_index(data)
    # Squared-L2 distances from FAISS; convert to Euclidean.
    sq_dists, _ = query_knn(index, data, k=k)
    dists = np.sqrt(np.maximum(sq_dists, 0.0))
    return _levina_bickel(dists)


def _levina_bickel(distances: np.ndarray) -> int:
    """Compute the Levina-Bickel MLE dimension from a (n, k) distance matrix.

    Column 0 is the self-distance (≈ 0) and is skipped.  The remaining k-1
    columns are the actual nearest-neighbour distances.
    """
    # Skip self (column 0) and work with the actual k-1 neighbours.
    dists = distances[:, 1:]
    if dists.shape[1] < 2:
        return 1
    # r_k: distance to the furthest neighbour used as the reference radius.
    r_k = dists[:, -1:]
    # r_j: distances to all closer neighbours.
    r_j = dists[:, :-1]
    # Clamp to avoid log(0); r_k/r_j >= 1 for sorted distances.
    log_ratios = np.log(np.maximum(r_k / np.maximum(r_j, 1e-10), 1e-10))
    mean_log = log_ratios.mean(axis=1)
    valid = mean_log > 1e-10
    if not np.any(valid):
        return 1
    dim_per_point = 1.0 / mean_log[valid]
    return max(1, int(np.round(np.median(dim_per_point))))


def _consensus_dimension(estimates: list[int], n_features: int) -> int:
    """Return median of *estimates*, clamped to [1, n_features - 1]."""
    median = int(np.round(np.median(estimates)))
    return max(1, min(median, max(1, n_features - 1)))


def _confidence_interval(estimates: list[int]) -> tuple[int, int]:
    """Return (min, max) of *estimates* as a simple confidence interval."""
    return int(min(estimates)), int(max(estimates))

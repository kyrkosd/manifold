"""
Feature graph construction for the Structure Discovery phase.
Combines absolute Pearson correlation, mutual information, and partial
correlation with adaptive thresholding to produce a robust feature-level graph.
NaN values in the input are imputed with 0 (the normalised column mean).
"""
from __future__ import annotations

import numpy as np
from sklearn.covariance import GraphicalLasso
from sklearn.feature_selection import mutual_info_regression

from common.graph_utils import connected_components
from common.logging import get_logger
from common.types import FeatureGraph

_log = get_logger(__name__)

# Adaptive threshold multiplier: threshold = mean + _SIGMA * std of edge weights.
_SIGMA = 1.5
# GraphicalLasso requires at least this many samples to be numerically stable.
_MIN_LASSO_SAMPLES = 10


def build(data: np.ndarray) -> FeatureGraph:
    """Build a global feature graph over the columns of *data*.

    Computes three complementary graphs (correlation, MI, partial correlation),
    combines them by element-wise mean, applies an adaptive threshold, and
    packages the result as a FeatureGraph.

    Parameters
    ----------
    data : (n, d) float64 array; NaN values are imputed with 0.

    Returns
    -------
    FeatureGraph : d nodes (one per feature), weighted edges, and graph stats.
    """
    # Impute NaN with 0 (the zero-mean column baseline after normalisation).
    clean = np.where(np.isnan(data), 0.0, data)
    corr = _correlation_graph(clean)
    mi = _mutual_information_graph(clean)
    partial = _partial_correlation_graph(clean)
    combined = _combine_graphs([corr, mi, partial])
    threshold = _adaptive_threshold(combined)
    adj = _threshold_graph(combined, threshold)
    _validate_graph(adj)
    stats = _compute_graph_stats(adj)
    nodes = [str(i) for i in range(data.shape[1])]
    return FeatureGraph(adjacency=adj, nodes=nodes, stats=stats)


def build_local(data: np.ndarray, region_mask: np.ndarray) -> FeatureGraph:
    """Build a feature graph from the rows of *data* selected by *region_mask*.

    Parameters
    ----------
    data : (n, d) float64 array.
    region_mask : (n,) boolean array; True selects a row for the local graph.

    Returns
    -------
    FeatureGraph : feature graph built exclusively from the masked subset.
    """
    local_data = data[region_mask]
    _log.debug("build_local: using %d of %d rows.", local_data.shape[0], data.shape[0])
    return build(local_data)


def _correlation_graph(data: np.ndarray) -> np.ndarray:
    """Return a (d, d) absolute Pearson correlation matrix with zero diagonal."""
    # corrcoef expects variables as rows, so transpose to put features on rows.
    corr = np.abs(np.corrcoef(data.T))
    np.fill_diagonal(corr, 0.0)
    # NaN arises when a feature has zero variance after imputation.
    return np.nan_to_num(corr, nan=0.0)


def _mutual_information_graph(data: np.ndarray) -> np.ndarray:
    """Return a (d, d) symmetric normalised pairwise mutual information matrix."""
    d = data.shape[1]
    mi = np.zeros((d, d))
    for j in range(d):
        target = data[:, j]
        # Skip constant targets; MI with a constant is always 0.
        if np.std(target) < 1e-10:
            continue
        mi[:, j] = mutual_info_regression(data, target, random_state=0)
    # Symmetrise and clear diagonal (self-MI is not an edge).
    mi = (mi + mi.T) / 2.0
    np.fill_diagonal(mi, 0.0)
    # Normalise to [0, 1] so MI is comparable to correlation weights.
    max_mi = mi.max()
    if max_mi > 1e-10:
        mi /= max_mi
    return mi


def _partial_correlation_graph(data: np.ndarray) -> np.ndarray:
    """Return a (d, d) absolute partial correlation matrix via the precision matrix.

    Tries GraphicalLasso for sparse precision estimation; falls back to the
    Moore–Penrose pseudoinverse of the sample covariance when Lasso fails.
    """
    n, d = data.shape
    try:
        if n < _MIN_LASSO_SAMPLES or d >= n:
            raise ValueError("Dataset too small for GraphicalLasso.")
        gl = GraphicalLasso(max_iter=200)
        gl.fit(data)
        precision = gl.precision_
    except Exception:
        _log.debug("GraphicalLasso failed; falling back to pseudoinverse.")
        cov = np.cov(data.T)
        precision = np.linalg.pinv(cov if cov.ndim == 2 else np.atleast_2d(cov))
    return _precision_to_partial_corr(precision)


def _precision_to_partial_corr(precision: np.ndarray) -> np.ndarray:
    """Convert a precision matrix to an absolute partial correlation matrix."""
    diag = np.sqrt(np.abs(np.diag(precision)))
    # Replace near-zero diagonal entries with 1.0 to avoid division by zero.
    diag = np.where(diag < 1e-10, 1.0, diag)
    partial = np.clip(np.abs(precision) / np.outer(diag, diag), 0.0, 1.0)
    np.fill_diagonal(partial, 0.0)
    return partial


def _adaptive_threshold(weights: np.ndarray) -> float:
    """Return mean + _SIGMA * std of the upper-triangle edge weights."""
    upper = weights[np.triu_indices_from(weights, k=1)]
    return float(np.mean(upper) + _SIGMA * np.std(upper))


def _combine_graphs(graphs: list[np.ndarray]) -> np.ndarray:
    """Return the element-wise mean of adjacency matrices in *graphs*."""
    return np.mean(np.stack(graphs, axis=0), axis=0)


def _threshold_graph(weights: np.ndarray, threshold: float) -> np.ndarray:
    """Zero out edges strictly below *threshold*; return a new array."""
    result = weights.copy()
    result[result < threshold] = 0.0
    return result


def _validate_graph(graph: np.ndarray) -> bool:
    """Return True if *graph* is symmetric, non-negative, and has zero diagonal."""
    symmetric = bool(np.allclose(graph, graph.T, atol=1e-8))
    non_negative = bool(np.all(graph >= -1e-10))
    zero_diag = bool(np.allclose(np.diag(graph), 0.0))
    valid = symmetric and non_negative and zero_diag
    if not valid:
        _log.warning(
            "Graph validation: symmetric=%s non_negative=%s zero_diag=%s",
            symmetric, non_negative, zero_diag,
        )
    return valid


def _compute_graph_stats(graph: np.ndarray) -> dict[str, object]:
    """Return a dict of n_edges, density, n_components, and avg_degree."""
    n = graph.shape[0]
    # Count only upper-triangle edges (undirected, no double-counting).
    n_edges = int(np.sum(graph[np.triu_indices_from(graph, k=1)] > 0))
    max_edges = n * (n - 1) / 2
    density = float(n_edges / max_edges) if max_edges > 0 else 0.0
    n_components = len(connected_components(graph))
    avg_degree = float(np.mean((graph > 0).sum(axis=1)))
    return {
        "n_edges": n_edges,
        "density": density,
        "n_components": n_components,
        "avg_degree": avg_degree,
    }

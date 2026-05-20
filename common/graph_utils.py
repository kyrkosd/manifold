"""
Graph construction and manipulation helpers: Laplacian builders, degree
normalisation, connected components, k-NN graphs, edge thresholding,
Dijkstra shortest paths, diameter, and clustering coefficient.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as _scipy_cc
from scipy.sparse.csgraph import dijkstra as _dijkstra
from sklearn.neighbors import NearestNeighbors


def laplacian(adjacency: np.ndarray) -> np.ndarray:
    """Compute the unnormalised graph Laplacian ``L = D − W``.

    Parameters
    ----------
    adjacency:
        (n, n) symmetric non-negative weight matrix.

    Returns
    -------
    np.ndarray
        (n, n) Laplacian; row sums are zero.
    """
    return degree_matrix(adjacency) - adjacency


def normalized_laplacian(adjacency: np.ndarray) -> np.ndarray:
    """Compute the symmetric normalised Laplacian ``L_sym = D^{-1/2} L D^{-1/2}``.

    Isolated nodes (degree zero) receive zero rows and columns.

    Parameters
    ----------
    adjacency:
        (n, n) symmetric non-negative weight matrix.

    Returns
    -------
    np.ndarray
        (n, n) normalised Laplacian.
    """
    d = adjacency.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, d ** -0.5, 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return D_inv_sqrt @ laplacian(adjacency) @ D_inv_sqrt


def degree_matrix(adjacency: np.ndarray) -> np.ndarray:
    """Compute the diagonal degree matrix from an adjacency/weight matrix.

    Parameters
    ----------
    adjacency:
        (n, n) weight matrix.

    Returns
    -------
    np.ndarray
        (n, n) diagonal matrix whose entries are the row sums of *adjacency*.
    """
    return np.diag(adjacency.sum(axis=1))


def connected_components(adjacency: np.ndarray) -> list[set[int]]:
    """Find the connected components of an undirected weighted graph.

    Parameters
    ----------
    adjacency:
        (n, n) weight matrix; non-zero entries indicate edges.

    Returns
    -------
    list[set[int]]
        One set of node indices per connected component.
    """
    n = adjacency.shape[0]
    if n == 0:
        return []
    sparse = csr_matrix(adjacency != 0, dtype=bool)
    n_components, labels = _scipy_cc(sparse, directed=False)
    return [
        {i for i, lbl in enumerate(labels) if lbl == c}
        for c in range(n_components)
    ]


def knn_graph(data: np.ndarray, k: int) -> np.ndarray:
    """Build a symmetric k-NN adjacency graph from row vectors in *data*.

    An edge exists between i and j if j ∈ k-NN(i) **or** i ∈ k-NN(j).

    Parameters
    ----------
    data:
        (n, d) feature matrix.
    k:
        Number of nearest neighbours (excluding self).

    Returns
    -------
    np.ndarray
        (n, n) symmetric binary adjacency matrix.

    Raises
    ------
    ValueError
        If *k* >= n.
    """
    n = data.shape[0]
    if k >= n:
        raise ValueError(f"k={k} must be strictly less than n={n}.")
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn.fit(data)
    indices = nn.kneighbors(data, return_distance=False)[:, 1:]
    adj = np.zeros((n, n), dtype=float)
    for i, neighbors in enumerate(indices):
        adj[i, neighbors] = 1.0
    return np.maximum(adj, adj.T)


def threshold_graph(weights: np.ndarray, threshold: float) -> np.ndarray:
    """Zero out edges whose weight is strictly below *threshold*.

    Parameters
    ----------
    weights:
        (n, n) edge weight matrix.
    threshold:
        Minimum weight to retain.

    Returns
    -------
    np.ndarray
        (n, n) thresholded weight matrix.  The original array is not modified.
    """
    result = weights.copy()
    result[result < threshold] = 0.0
    return result


def dijkstra(graph: np.ndarray, start: int) -> np.ndarray:
    """Compute shortest-path distances from *start* to every other node.

    Uses scipy's optimised Dijkstra implementation on a sparse representation.

    Parameters
    ----------
    graph:
        (n, n) non-negative weight matrix; 0 means no edge.
    start:
        Source node index.

    Returns
    -------
    np.ndarray
        (n,) shortest-path distances from *start*; ``inf`` for unreachable nodes.
    """
    sparse = csr_matrix(graph)
    return _dijkstra(sparse, indices=start)


def graph_diameter(graph: np.ndarray) -> float:
    """Compute the diameter (maximum finite shortest-path distance) of the graph.

    Parameters
    ----------
    graph:
        (n, n) non-negative weight matrix.

    Returns
    -------
    float
        Maximum finite pairwise shortest-path distance, or ``inf`` if the
        graph is disconnected and no finite distances exist.
    """
    sparse = csr_matrix(graph)
    all_distances = _dijkstra(sparse)
    finite = all_distances[np.isfinite(all_distances)]
    return float(np.max(finite)) if len(finite) > 0 else float("inf")


def clustering_coefficient(adjacency: np.ndarray) -> float:
    """Compute the mean local clustering coefficient of the graph.

    Uses the standard triangle-counting formula:
    cc_i = A³[i,i] / (k_i × (k_i − 1)), averaged over all nodes.

    Parameters
    ----------
    adjacency:
        (n, n) weight matrix; binarised internally.

    Returns
    -------
    float
        Mean local clustering coefficient in [0, 1].
        Isolated nodes (degree < 2) contribute 0.
    """
    binary = (adjacency > 0).astype(float)
    np.fill_diagonal(binary, 0)
    triangles = np.diagonal(binary @ binary @ binary)
    degrees = binary.sum(axis=1)
    possible = degrees * (degrees - 1)
    local_cc = np.where(possible > 0, triangles / possible, 0.0)
    return float(np.mean(local_cc))

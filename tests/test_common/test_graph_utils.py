"""Tests for common/graph_utils.py."""
from __future__ import annotations

import numpy as np
import pytest

from common.graph_utils import (
    clustering_coefficient,
    connected_components,
    degree_matrix,
    dijkstra,
    graph_diameter,
    knn_graph,
    laplacian,
    normalized_laplacian,
    threshold_graph,
)


# --- laplacian ---


def test_laplacian_row_sums_zero() -> None:
    """Laplacian row sums zero."""
    adj = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    np.testing.assert_allclose(laplacian(adj).sum(axis=1), np.zeros(3), atol=1e-10)


def test_laplacian_diagonal_equals_degree() -> None:
    """Laplacian diagonal equals degree."""
    adj = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    np.testing.assert_allclose(np.diag(laplacian(adj)), [1.0, 2.0, 1.0], atol=1e-10)


def test_laplacian_empty_adjacency_is_zero() -> None:
    """Laplacian empty adjacency is zero."""
    np.testing.assert_allclose(laplacian(np.zeros((3, 3))), np.zeros((3, 3)), atol=1e-10)


def test_laplacian_single_node() -> None:
    """Laplacian single node."""
    np.testing.assert_allclose(laplacian(np.array([[0.0]])), [[0.0]], atol=1e-10)


# --- normalized_laplacian ---


def test_normalized_laplacian_symmetric() -> None:
    """Normalized laplacian symmetric."""
    adj = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    norm_lap = normalized_laplacian(adj)
    np.testing.assert_allclose(norm_lap, norm_lap.T, atol=1e-10)


def test_normalized_laplacian_eigenvalues_in_range() -> None:
    """Normalized laplacian eigenvalues in range."""
    adj = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    eigvals = np.linalg.eigvalsh(normalized_laplacian(adj))
    assert np.all(eigvals >= -1e-10)
    assert np.all(eigvals <= 2.0 + 1e-10)


def test_normalized_laplacian_isolated_node_zero_row_col() -> None:
    """Normalized laplacian isolated node zero row col."""
    adj = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    norm_lap = normalized_laplacian(adj)
    np.testing.assert_allclose(norm_lap[0, :], np.zeros(3), atol=1e-10)
    np.testing.assert_allclose(norm_lap[:, 0], np.zeros(3), atol=1e-10)


# --- degree_matrix ---


def test_degree_matrix_diagonal() -> None:
    """Degree matrix diagonal."""
    adj = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(np.diag(degree_matrix(adj)), [2.0, 1.0, 1.0], atol=1e-10)


def test_degree_matrix_off_diagonal_zero() -> None:
    """Degree matrix off diagonal zero."""
    deg_mat = degree_matrix(np.ones((4, 4)))
    np.testing.assert_allclose(deg_mat - np.diag(np.diag(deg_mat)), np.zeros((4, 4)), atol=1e-10)


def test_degree_matrix_all_zeros() -> None:
    """Degree matrix all zeros."""
    np.testing.assert_allclose(degree_matrix(np.zeros((4, 4))), np.zeros((4, 4)), atol=1e-10)


# --- connected_components ---


def test_connected_components_fully_connected() -> None:
    """Connected components fully connected."""
    adj = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    components = connected_components(adj)
    assert len(components) == 1
    assert components[0] == {0, 1, 2}


def test_connected_components_two_isolated_pairs() -> None:
    """Connected components two isolated pairs."""
    adj = np.zeros((4, 4))
    adj[0, 1] = adj[1, 0] = 1.0
    adj[2, 3] = adj[3, 2] = 1.0
    components = connected_components(adj)
    assert len(components) == 2
    sets = [frozenset(c) for c in components]
    assert frozenset({0, 1}) in sets
    assert frozenset({2, 3}) in sets


def test_connected_components_all_isolated() -> None:
    """Connected components all isolated."""
    assert len(connected_components(np.zeros((3, 3)))) == 3


def test_connected_components_single_node() -> None:
    """Connected components single node."""
    assert 0 in connected_components(np.array([[0.0]]))[0]


def test_connected_components_empty_graph() -> None:
    """Connected components empty graph."""
    assert connected_components(np.zeros((0, 0))) == []


# --- knn_graph ---


def test_knn_graph_symmetric() -> None:
    """Knn graph symmetric."""
    data = np.random.default_rng(1).standard_normal((10, 2))
    adj = knn_graph(data, k=3)
    np.testing.assert_allclose(adj, adj.T, atol=1e-10)


def test_knn_graph_no_self_loops() -> None:
    """Knn graph no self loops."""
    data = np.random.default_rng(2).standard_normal((6, 2))
    np.testing.assert_allclose(np.diag(knn_graph(data, k=2)), np.zeros(6), atol=1e-10)


def test_knn_graph_k_equals_n_raises() -> None:
    """Knn graph k equals n raises."""
    with pytest.raises(ValueError):
        knn_graph(np.array([[1.0], [2.0]]), k=2)


def test_knn_graph_collinear_k1() -> None:
    """Knn graph collinear k1."""
    data = np.array([[0.0], [1.0], [2.0], [10.0]])
    adj = knn_graph(data, k=1)
    assert adj[0, 1] == 1.0 or adj[1, 0] == 1.0


# --- threshold_graph ---


def test_threshold_graph_removes_weak_edges() -> None:
    """Threshold graph removes weak edges."""
    weight_mat = np.array([[0.0, 0.3, 0.8], [0.3, 0.0, 0.5], [0.8, 0.5, 0.0]])
    result = threshold_graph(weight_mat, threshold=0.5)
    assert result[0, 1] == pytest.approx(0.0)
    assert result[0, 2] == pytest.approx(0.8)


def test_threshold_graph_boundary_retained() -> None:
    """Threshold graph boundary retained."""
    # 0.5 is NOT strictly below 0.5 → kept
    result = threshold_graph(np.array([[0.0, 0.5], [0.5, 0.0]]), threshold=0.5)
    assert result[0, 1] == pytest.approx(0.5)


def test_threshold_graph_does_not_mutate_input() -> None:
    """Threshold graph does not mutate input."""
    weight_mat = np.array([[0.0, 0.5], [0.5, 0.0]])
    original = weight_mat.copy()
    threshold_graph(weight_mat, threshold=1.0)
    np.testing.assert_allclose(weight_mat, original, atol=1e-10)


# --- dijkstra ---


def test_dijkstra_known_distances() -> None:
    """Dijkstra known distances."""
    # Linear chain: 0-1-2-3 with unit weights
    g = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    d = dijkstra(g, start=0)
    np.testing.assert_allclose(d, [0.0, 1.0, 2.0, 3.0], atol=1e-10)


def test_dijkstra_self_distance_zero() -> None:
    """Dijkstra self distance zero."""
    g = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert dijkstra(g, start=0)[0] == pytest.approx(0.0)


def test_dijkstra_disconnected_node_inf() -> None:
    """Dijkstra disconnected node inf."""
    g = np.zeros((3, 3))
    g[0, 1] = g[1, 0] = 1.0
    d = dijkstra(g, start=0)
    assert np.isinf(d[2])


def test_dijkstra_returns_array_of_length_n() -> None:
    """Dijkstra returns array of length n."""
    g = np.ones((5, 5)) - np.eye(5)
    assert dijkstra(g, start=2).shape == (5,)


# --- graph_diameter ---


def test_graph_diameter_linear_chain() -> None:
    """Graph diameter linear chain."""
    g = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    assert graph_diameter(g) == pytest.approx(3.0)


def test_graph_diameter_complete_graph() -> None:
    """Graph diameter complete graph."""
    g = np.ones((4, 4)) - np.eye(4)
    assert graph_diameter(g) == pytest.approx(1.0)


def test_graph_diameter_disconnected_is_inf() -> None:
    """Graph diameter disconnected is inf."""
    g = np.zeros((4, 4))
    g[0, 1] = g[1, 0] = 1.0
    # nodes 2 and 3 are isolated → diameter is inf
    assert np.isinf(graph_diameter(g))


# --- clustering_coefficient ---


def test_clustering_coefficient_complete_triangle() -> None:
    """Clustering coefficient complete triangle."""
    # Complete graph on 3 nodes → every node forms a triangle → cc = 1.0
    g = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    assert clustering_coefficient(g) == pytest.approx(1.0)


def test_clustering_coefficient_star_graph() -> None:
    """Clustering coefficient star graph."""
    # Star graph: hub connects to 3 leaves; no triangles → cc = 0
    g = np.zeros((4, 4))
    for i in range(1, 4):
        g[0, i] = g[i, 0] = 1.0
    assert clustering_coefficient(g) == pytest.approx(0.0)


def test_clustering_coefficient_isolated_nodes_zero() -> None:
    """Clustering coefficient isolated nodes zero."""
    # All isolated nodes contribute 0
    assert clustering_coefficient(np.zeros((5, 5))) == pytest.approx(0.0)


def test_clustering_coefficient_range() -> None:
    """Clustering coefficient range."""
    data = np.random.default_rng(0).standard_normal((10, 2))
    g = knn_graph(data, k=3)
    cc = clustering_coefficient(g)
    assert 0.0 <= cc <= 1.0

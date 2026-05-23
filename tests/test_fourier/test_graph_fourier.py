"""Tests for fourier/graph_fourier.py.

Known graph Laplacian eigenvalues used to anchor numerical assertions:
  K4 (complete 4-node):  [0, 4, 4, 4]
  P4 (path 0-1-2-3):    [0, 2-√2, 2, 2+√2]
"""
from __future__ import annotations

import numpy as np
import pytest

from common.types import FeatureGraph
from fourier.graph_fourier import GraphFourierEngine


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _k4_graph() -> FeatureGraph:
    """Complete K4 graph; Laplacian eigenvalues are [0, 4, 4, 4]."""
    adj = np.ones((4, 4)) - np.eye(4)
    return FeatureGraph(adjacency=adj, nodes=["0", "1", "2", "3"], stats={})


def _p4_graph() -> FeatureGraph:
    """Path graph 0-1-2-3; eigenvalues [0, 2-√2, 2, 2+√2]."""
    adj = np.zeros((4, 4))
    for i, j in [(0, 1), (1, 2), (2, 3)]:
        adj[i, j] = adj[j, i] = 1.0
    return FeatureGraph(adjacency=adj, nodes=["0", "1", "2", "3"], stats={})


def _data(n: int = 20, d: int = 4, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, d))


# ---------------------------------------------------------------------------
# GraphFourierEngine.__init__ — eigenbasis properties
# ---------------------------------------------------------------------------

class TestGraphFourierInit:
    """Tests for Graph Fourier Init."""
    def test_eigenvectors_shape(self):
        # K4 has 4 nodes; eigenvector matrix must be square (4, 4).
        engine = GraphFourierEngine(_k4_graph())
        assert engine.basis.eigenvectors.shape == (4, 4)

    def test_eigenvalues_shape(self):
        # One eigenvalue per spectral component.
        engine = GraphFourierEngine(_k4_graph())
        assert engine.basis.eigenvalues.shape == (4,)

    def test_eigenvalues_sorted_ascending(self):
        # EigenBasis contract: eigenvalues are non-decreasing.
        engine = GraphFourierEngine(_k4_graph())
        vals = engine.basis.eigenvalues
        assert np.all(vals[1:] >= vals[:-1] - 1e-10)

    def test_k4_smallest_eigenvalue_zero(self):
        # Laplacian is positive semi-definite; first eigenvalue is 0 (dc mode).
        engine = GraphFourierEngine(_k4_graph())
        assert engine.basis.eigenvalues[0] == pytest.approx(0.0, abs=1e-8)

    def test_k4_nonzero_eigenvalues_are_four(self):
        # K4 has eigenvalue 4 with multiplicity 3.
        engine = GraphFourierEngine(_k4_graph())
        np.testing.assert_allclose(engine.basis.eigenvalues[1:], 4.0, atol=1e-8)

    def test_eigenvectors_orthonormal(self):
        # eigenvec_mat^T @ eigenvec_mat = I; orthonormality guarantees an exact round-trip.
        engine = GraphFourierEngine(_p4_graph())
        eigenvec_mat = engine.basis.eigenvectors
        np.testing.assert_allclose(eigenvec_mat.T @ eigenvec_mat, np.eye(4), atol=1e-10)


# ---------------------------------------------------------------------------
# transform / inverse_transform — shape and round-trip
# ---------------------------------------------------------------------------

class TestTransformAndInverse:
    """Tests for Transform And Inverse."""
    def test_transform_shape(self):
        # Forward GFT must preserve the (n, d) data shape.
        engine = GraphFourierEngine(_k4_graph())
        assert engine.transform(_data(n=10, d=4)).shape == (10, 4)

    def test_inverse_shape(self):
        # Inverse GFT must also return (n, d).
        engine = GraphFourierEngine(_k4_graph())
        assert engine.inverse_transform(_data(n=10, d=4, seed=1)).shape == (10, 4)

    def test_round_trip_recovery(self):
        # forward followed by inverse must recover the original data exactly.
        engine = GraphFourierEngine(_p4_graph())
        data = _data(n=15, d=4, seed=2)
        recovered = engine.inverse_transform(engine.transform(data))
        np.testing.assert_allclose(recovered, data, atol=1e-10)

    def test_transform_local_shape(self):
        # transform_local output rows == number of True entries in mask.
        engine = GraphFourierEngine(_k4_graph())
        data = _data(n=20, d=4, seed=3)
        mask = np.zeros(20, dtype=bool)
        mask[:8] = True
        assert engine.transform_local(data, mask).shape == (8, 4)

    def test_transform_local_all_true_matches_full(self):
        # All-True mask: transform_local must equal transform on the same data.
        engine = GraphFourierEngine(_k4_graph())
        data = _data(n=12, d=4, seed=4)
        mask = np.ones(12, dtype=bool)
        np.testing.assert_allclose(
            engine.transform_local(data, mask),
            engine.transform(data),
            atol=1e-12,
        )


# ---------------------------------------------------------------------------
# Disconnected graph handling
# ---------------------------------------------------------------------------

class TestDisconnectedGraph:
    """Tests for Disconnected Graph."""
    def test_all_zero_adjacency_does_not_raise(self):
        # Fully disconnected graph: all-zero Laplacian, all-zero eigenvalues.
        adj = np.zeros((4, 4))
        graph = FeatureGraph(adjacency=adj, nodes=["0", "1", "2", "3"], stats={})
        engine = GraphFourierEngine(graph)
        assert engine.basis.eigenvectors.shape == (4, 4)

    def test_round_trip_with_partial_graph(self):
        # Only one edge (0-1); nodes 2 and 3 are isolated — round-trip must hold.
        adj = np.zeros((4, 4))
        adj[0, 1] = adj[1, 0] = 1.0
        graph = FeatureGraph(adjacency=adj, nodes=["0", "1", "2", "3"], stats={})
        engine = GraphFourierEngine(graph)
        data = _data(n=10, d=4, seed=5)
        recovered = engine.inverse_transform(engine.transform(data))
        np.testing.assert_allclose(recovered, data, atol=1e-10)

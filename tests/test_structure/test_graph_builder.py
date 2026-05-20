"""Tests for structure/graph_builder.py.

The graph is over features (columns), so adjacency shape is (d, d).
Each test fixture is documented with its intended structure so that
assertion thresholds are easy to justify.
"""
from __future__ import annotations

import numpy as np
import pytest

from common.types import FeatureGraph
from structure.graph_builder import (
    _adaptive_threshold,
    _combine_graphs,
    _compute_graph_stats,
    _correlation_graph,
    _mutual_information_graph,
    _partial_correlation_graph,
    _precision_to_partial_corr,
    _threshold_graph,
    _validate_graph,
    build,
    build_local,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _independent_data(n: int = 60, d: int = 6, seed: int = 0) -> np.ndarray:
    """Return n×d data with statistically independent columns."""
    return np.random.default_rng(seed).standard_normal((n, d))


def _correlated_data(n: int = 80, seed: int = 1) -> np.ndarray:
    """Return n×6 data with three pairs of highly correlated features.

    Features 0-1 share signal base0, features 2-3 share base1, and
    features 4-5 share base2, each with small independent noise.
    """
    rng = np.random.default_rng(seed)
    noise_scale = 0.05
    bases = [rng.standard_normal(n) for _ in range(3)]
    noise = rng.standard_normal((n, 6)) * noise_scale
    return np.column_stack([
        bases[0] + noise[:, 0], bases[0] + noise[:, 1],
        bases[1] + noise[:, 2], bases[1] + noise[:, 3],
        bases[2] + noise[:, 4], bases[2] + noise[:, 5],
    ])


# ---------------------------------------------------------------------------
# build()
# ---------------------------------------------------------------------------

class TestBuild:
    def test_returns_feature_graph(self):
        # The return type must always be FeatureGraph.
        result = build(_independent_data())
        assert isinstance(result, FeatureGraph)

    def test_adjacency_shape_is_d_by_d(self):
        # Graph nodes are features (columns), not samples.
        data = _independent_data(n=50, d=8)
        result = build(data)
        assert result.adjacency.shape == (8, 8)

    def test_adjacency_is_symmetric(self):
        # The feature graph is undirected; adjacency must be symmetric.
        result = build(_independent_data())
        np.testing.assert_allclose(
            result.adjacency, result.adjacency.T, atol=1e-8
        )

    def test_adjacency_zero_diagonal(self):
        # No self-loops: every node has zero weight to itself.
        result = build(_independent_data())
        np.testing.assert_allclose(np.diag(result.adjacency), 0.0, atol=1e-10)

    def test_nodes_match_feature_count(self):
        # Number of node labels must equal d.
        data = _independent_data(n=50, d=5)
        result = build(data)
        assert len(result.nodes) == 5

    def test_nodes_are_string_feature_indices(self):
        # Node identifiers are the string representations of column indices.
        data = _independent_data(n=50, d=4)
        result = build(data)
        assert result.nodes == ["0", "1", "2", "3"]

    def test_stats_keys_present(self):
        # All four stats keys must be present regardless of graph structure.
        result = build(_independent_data())
        for key in ("n_edges", "density", "n_components", "avg_degree"):
            assert key in result.stats, f"Missing stat key: {key}"

    def test_adjacency_non_negative(self):
        # Edge weights must be non-negative after thresholding.
        result = build(_independent_data())
        assert np.all(result.adjacency >= 0)

    def test_nan_in_data_does_not_crash(self):
        # NaN values are imputed; the function must not raise.
        data = _independent_data()
        data[0, 0] = np.nan
        result = build(data)
        assert isinstance(result, FeatureGraph)

    def test_correlated_data_has_edges(self):
        # Highly correlated feature pairs should produce at least some edges.
        result = build(_correlated_data())
        assert result.stats["n_edges"] > 0


# ---------------------------------------------------------------------------
# build_local()
# ---------------------------------------------------------------------------

class TestBuildLocal:
    def test_returns_feature_graph(self):
        data = _independent_data(n=80, d=6)
        mask = np.zeros(80, dtype=bool)
        mask[:40] = True   # select first half of rows
        result = build_local(data, mask)
        assert isinstance(result, FeatureGraph)

    def test_adjacency_shape_unchanged(self):
        # d (columns) is fixed; shape of adjacency does not depend on mask.
        data = _independent_data(n=80, d=6)
        mask = np.ones(80, dtype=bool)
        mask[60:] = False   # drop last 20 rows
        result = build_local(data, mask)
        assert result.adjacency.shape == (6, 6)

    def test_mask_selects_subset(self):
        # build_local with all-True mask should equal build().
        data = _independent_data(n=60, d=5, seed=7)
        mask = np.ones(60, dtype=bool)
        full = build(data)
        local = build_local(data, mask)
        np.testing.assert_allclose(full.adjacency, local.adjacency, atol=1e-10)


# ---------------------------------------------------------------------------
# _correlation_graph
# ---------------------------------------------------------------------------

class TestCorrelationGraph:
    def test_shape(self):
        data = _independent_data(n=50, d=4)
        result = _correlation_graph(data)
        assert result.shape == (4, 4)

    def test_zero_diagonal(self):
        # Self-correlations are excluded from the graph.
        data = _independent_data()
        result = _correlation_graph(data)
        np.testing.assert_allclose(np.diag(result), 0.0)

    def test_values_in_unit_interval(self):
        # Absolute correlation is in [0, 1].
        data = _independent_data()
        result = _correlation_graph(data)
        assert np.all(result >= 0) and np.all(result <= 1.0 + 1e-8)

    def test_symmetric(self):
        data = _independent_data()
        result = _correlation_graph(data)
        np.testing.assert_allclose(result, result.T, atol=1e-10)

    def test_perfectly_correlated_features_near_one(self):
        # Two identical features must have |corr| ≈ 1.
        rng = np.random.default_rng(10)
        base = rng.standard_normal(50)
        data = np.column_stack([base, base, rng.standard_normal(50)])
        result = _correlation_graph(data)
        assert result[0, 1] > 0.99


# ---------------------------------------------------------------------------
# _mutual_information_graph
# ---------------------------------------------------------------------------

class TestMIGraph:
    def test_shape(self):
        data = _independent_data(n=50, d=5)
        result = _mutual_information_graph(data)
        assert result.shape == (5, 5)

    def test_zero_diagonal(self):
        data = _independent_data()
        result = _mutual_information_graph(data)
        np.testing.assert_allclose(np.diag(result), 0.0)

    def test_symmetric(self):
        data = _independent_data()
        result = _mutual_information_graph(data)
        np.testing.assert_allclose(result, result.T, atol=1e-8)

    def test_values_in_unit_interval(self):
        # MI is normalised to [0, 1] by dividing by the maximum.
        data = _independent_data()
        result = _mutual_information_graph(data)
        assert np.all(result >= 0) and np.all(result <= 1.0 + 1e-8)

    def test_constant_feature_gives_zero_column(self):
        # Constant feature has zero MI with everything; column must be zero.
        rng = np.random.default_rng(5)
        data = rng.standard_normal((50, 3))
        data[:, 1] = 5.0   # constant column
        result = _mutual_information_graph(data)
        np.testing.assert_allclose(result[:, 1], 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# _partial_correlation_graph and _precision_to_partial_corr
# ---------------------------------------------------------------------------

class TestPartialCorrelationGraph:
    def test_shape(self):
        data = _independent_data(n=80, d=5)
        result = _partial_correlation_graph(data)
        assert result.shape == (5, 5)

    def test_zero_diagonal(self):
        data = _independent_data(n=80, d=5)
        result = _partial_correlation_graph(data)
        np.testing.assert_allclose(np.diag(result), 0.0)

    def test_values_in_unit_interval(self):
        # Absolute partial correlation is in [0, 1].
        data = _independent_data(n=80, d=5)
        result = _partial_correlation_graph(data)
        assert np.all(result >= 0) and np.all(result <= 1.0 + 1e-8)

    def test_small_data_falls_back_gracefully(self):
        # With fewer samples than features, GraphicalLasso fails; pseudoinverse kicks in.
        data = _independent_data(n=8, d=6)
        result = _partial_correlation_graph(data)  # must not raise
        assert result.shape == (6, 6)


class TestPrecisionToPartialCorr:
    def test_identity_precision_gives_zero_partial_corr(self):
        # Identity precision matrix → no off-diagonal partial correlation.
        P = np.eye(4)
        result = _precision_to_partial_corr(P)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_output_clipped_to_unit_interval(self):
        # Even with a numerically ill-conditioned matrix, output must be in [0, 1].
        P = np.full((3, 3), 100.0)
        np.fill_diagonal(P, 200.0)
        result = _precision_to_partial_corr(P)
        assert np.all(result >= 0) and np.all(result <= 1.0 + 1e-8)


# ---------------------------------------------------------------------------
# Thresholding and utility helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_adaptive_threshold_formula(self):
        # Known matrix: upper triangle = [0.1, 0.9]; mean=0.5, std≈0.4.
        w = np.array([[0.0, 0.1], [0.1, 0.0]])
        w2 = np.array([[0.0, 0.9], [0.9, 0.0]])
        combined = np.array([[0.0, 0.5], [0.5, 0.0]])  # mean of above
        upper = np.array([0.5])
        expected = float(np.mean(upper) + 1.5 * np.std(upper))
        result = _adaptive_threshold(combined)
        assert result == pytest.approx(expected)

    def test_combine_graphs_elementwise_mean(self):
        # Three (2,2) matrices → element-wise mean.
        a = np.array([[0.0, 0.3], [0.3, 0.0]])
        b = np.array([[0.0, 0.6], [0.6, 0.0]])
        c = np.array([[0.0, 0.9], [0.9, 0.0]])
        result = _combine_graphs([a, b, c])
        np.testing.assert_allclose(result, np.array([[0.0, 0.6], [0.6, 0.0]]))

    def test_threshold_graph_zeros_below(self):
        # Edges with weight < threshold must become 0.
        w = np.array([[0.0, 0.2, 0.8], [0.2, 0.0, 0.5], [0.8, 0.5, 0.0]])
        result = _threshold_graph(w, 0.6)
        assert result[0, 1] == pytest.approx(0.0)   # 0.2 < 0.6 → zeroed
        assert result[0, 2] == pytest.approx(0.8)   # 0.8 >= 0.6 → kept

    def test_threshold_graph_does_not_modify_original(self):
        # _threshold_graph must return a copy, not modify in-place.
        w = np.array([[0.0, 0.3], [0.3, 0.0]])
        original = w.copy()
        _threshold_graph(w, 0.5)
        np.testing.assert_array_equal(w, original)

    def test_validate_graph_valid(self):
        # A symmetric, non-negative, zero-diagonal matrix is valid.
        g = np.array([[0.0, 0.5], [0.5, 0.0]])
        assert _validate_graph(g) is True

    def test_validate_graph_non_symmetric(self):
        # Asymmetric matrix must fail validation.
        g = np.array([[0.0, 0.3], [0.7, 0.0]])
        assert _validate_graph(g) is False

    def test_validate_graph_non_zero_diagonal(self):
        # A self-loop (non-zero diagonal) must fail validation.
        g = np.array([[1.0, 0.5], [0.5, 1.0]])
        assert _validate_graph(g) is False

    def test_compute_graph_stats_keys(self):
        g = np.array([[0.0, 0.7, 0.0], [0.7, 0.0, 0.4], [0.0, 0.4, 0.0]])
        stats = _compute_graph_stats(g)
        assert set(stats.keys()) == {"n_edges", "density", "n_components", "avg_degree"}

    def test_compute_graph_stats_values(self):
        # 3-node path graph: 2 edges, density=2/3, 1 component.
        g = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        stats = _compute_graph_stats(g)
        assert stats["n_edges"] == 2
        assert stats["density"] == pytest.approx(2 / 3)
        assert stats["n_components"] == 1

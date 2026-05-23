"""Tests for common/nn_utils.py."""
from __future__ import annotations

import numpy as np
import pytest

from common.nn_utils import (
    batch_knn,
    build_faiss_index,
    query_knn,
    query_radius,
    rebuild_index,
)

# Fixed small dataset used across most tests
_RNG = np.random.default_rng(42)
_DATA = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0]], dtype=np.float32)


# --- build_faiss_index ---


def test_build_index_ntotal_matches_input() -> None:
    """Build index ntotal matches input."""
    index = build_faiss_index(_DATA)
    assert index.ntotal == len(_DATA)


def test_build_index_unsupported_metric_raises() -> None:
    """Build index unsupported metric raises."""
    with pytest.raises(ValueError, match="L2"):
        build_faiss_index(_DATA, metric="cosine")


def test_build_index_float64_input_accepted() -> None:
    """Build index float64 input accepted."""
    data_f64 = _DATA.astype(np.float64)
    index = build_faiss_index(data_f64)
    assert index.ntotal == len(_DATA)


def test_build_index_single_point() -> None:
    """Build index single point."""
    index = build_faiss_index(np.array([[1.0, 2.0]], dtype=np.float32))
    assert index.ntotal == 1


# --- query_knn ---


def test_query_knn_output_shapes() -> None:
    """Query knn output shapes."""
    index = build_faiss_index(_DATA)
    dists, idxs = query_knn(index, _DATA[:2], k=2)
    assert dists.shape == (2, 2)
    assert idxs.shape == (2, 2)


def test_query_knn_self_is_nearest_neighbour() -> None:
    """Query knn self is nearest neighbour."""
    index = build_faiss_index(_DATA)
    dists, idxs = query_knn(index, _DATA, k=1)
    np.testing.assert_allclose(dists[:, 0], np.zeros(len(_DATA)), atol=1e-5)
    np.testing.assert_array_equal(idxs[:, 0], np.arange(len(_DATA)))


def test_query_knn_distances_non_negative() -> None:
    """Query knn distances non negative."""
    index = build_faiss_index(_DATA)
    dists, _ = query_knn(index, _DATA, k=3)
    assert np.all(dists >= 0)


def test_query_knn_single_query_point() -> None:
    """Query knn single query point."""
    index = build_faiss_index(_DATA)
    dists, idxs = query_knn(index, _DATA[:1], k=2)
    assert dists.shape == (1, 2)
    assert idxs.shape == (1, 2)


# --- query_radius ---


def test_query_radius_returns_list_of_length_n_queries() -> None:
    """Query radius returns list of length n queries."""
    index = build_faiss_index(_DATA)
    results = query_radius(index, _DATA[:2], radius=1.5)
    assert isinstance(results, list)
    assert len(results) == 2


def test_query_radius_includes_self() -> None:
    """Query radius includes self."""
    index = build_faiss_index(_DATA)
    results = query_radius(index, _DATA[:1], radius=0.1)
    assert 0 in results[0].tolist()


def test_query_radius_origin_neighbours() -> None:
    """Query radius origin neighbours."""
    index = build_faiss_index(_DATA)
    query = np.array([[0.0, 0.0]], dtype=np.float32)
    results = query_radius(index, query, radius=1.1)
    indices = set(results[0].tolist())
    # [0,0], [1,0], [0,1] are all within Euclidean distance 1.1 of origin
    assert {0, 1, 2}.issubset(indices)
    # [2,2] is at distance sqrt(8) ≈ 2.83 — should not appear
    assert 3 not in indices


def test_query_radius_zero_radius_returns_only_self() -> None:
    """Query radius zero radius returns only self."""
    index = build_faiss_index(_DATA)
    results = query_radius(index, _DATA[:1], radius=0.0)
    # Only exact matches (self) within radius 0
    assert 0 in results[0].tolist()
    assert len(results[0]) == 1


# --- batch_knn ---


def test_batch_knn_matches_full_query_knn() -> None:
    """Batch knn matches full query knn."""
    index = build_faiss_index(_DATA)
    dists_batch, idxs_batch = batch_knn(index, _DATA, k=2, batch_size=2)
    dists_full, idxs_full = query_knn(index, _DATA, k=2)
    np.testing.assert_allclose(dists_batch, dists_full, atol=1e-5)
    np.testing.assert_array_equal(idxs_batch, idxs_full)


def test_batch_knn_output_shapes() -> None:
    """Batch knn output shapes."""
    index = build_faiss_index(_DATA)
    dists, idxs = batch_knn(index, _DATA, k=3, batch_size=1)
    assert dists.shape == (len(_DATA), 3)
    assert idxs.shape == (len(_DATA), 3)


def test_batch_knn_batch_larger_than_data() -> None:
    """Batch knn batch larger than data."""
    index = build_faiss_index(_DATA)
    dists, _ = batch_knn(index, _DATA, k=2, batch_size=10_000)
    assert dists.shape == (len(_DATA), 2)


# --- rebuild_index ---


def test_rebuild_index_ntotal_updated() -> None:
    """Rebuild index ntotal updated."""
    index = build_faiss_index(_DATA)
    new_data = np.array([[5.0, 5.0], [6.0, 6.0]], dtype=np.float32)
    index = rebuild_index(index, new_data)
    assert index.ntotal == 2


def test_rebuild_index_queries_new_data() -> None:
    """Rebuild index queries new data."""
    index = build_faiss_index(_DATA)
    new_data = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32)
    index = rebuild_index(index, new_data)
    dists, idxs = query_knn(index, new_data[:1], k=1)
    np.testing.assert_allclose(dists[0, 0], 0.0, atol=1e-5)
    assert idxs[0, 0] == 0


def test_rebuild_index_old_data_not_found() -> None:
    """Rebuild index old data not found."""
    index = build_faiss_index(_DATA)
    new_data = np.array([[100.0, 100.0]], dtype=np.float32)
    index = rebuild_index(index, new_data)
    # Original points are gone; nearest neighbour of [0,0] is now [100,100]
    query = np.array([[0.0, 0.0]], dtype=np.float32)
    dists, idxs = query_knn(index, query, k=1)
    assert idxs[0, 0] == 0  # only one point in new index
    assert dists[0, 0] > 100.0  # far from origin

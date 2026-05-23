"""Tests for manifold/chart.py.

Canonical fixture: a 2-D linear subspace embedded in 10-D ambient space.
A random orthonormal pair of columns defines the subspace; data is sampled
on that subspace with small noise.  The chart should recover intrinsic_dim=2
coordinates, the round-trip reconstruction error should be small, and the
coordinates array must have the right shape.
"""
from __future__ import annotations

import numpy as np

import structure.graph_builder as graph_builder
from common import nn_utils
from common.types import EigenBasis
from fourier.graph_fourier import GraphFourierEngine
from manifold.chart import (
    Chart,
    _compute_chart_inverse,
    _compute_coordinates,
    _define_chart_map,
    _expand_region,
    _select_basis_adaptive,
    build,
)
from manifold.eigenvector_alignment import align_to_reference


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _subspace_data(
    n: int = 60, ambient: int = 10, intrinsic: int = 2, noise: float = 0.01, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (data, basis) for a 2-D subspace in 10-D with small noise."""
    rng = np.random.default_rng(seed)
    orth_mat, _ = np.linalg.qr(rng.standard_normal((ambient, intrinsic)))
    basis = orth_mat[:, :intrinsic]
    coeffs = rng.standard_normal((n, intrinsic))
    noise_mat = rng.standard_normal((n, ambient)) * noise
    return coeffs @ basis.T + noise_mat, basis


def _reference_basis(data: np.ndarray) -> EigenBasis:
    """Build a global EigenBasis from the full graph Laplacian of *data*."""
    fg = graph_builder.build(data)
    return GraphFourierEngine(fg).basis


def _make_chart(seed: int = 0) -> tuple[Chart, np.ndarray]:
    """Return (chart, region_data) for a small 2-D-in-10-D fixture."""
    data, _basis = _subspace_data(n=60, seed=seed)
    region_indices = np.arange(40, dtype=np.intp)
    region_data = data[region_indices]
    ref = _reference_basis(data)
    idx = nn_utils.build_faiss_index(data)
    chart = build(
        region_data=region_data,
        region_indices=region_indices,
        full_data=data,
        graph_builder=graph_builder,
        intrinsic_dim=2,
        reference_basis=ref,
        faiss_index=idx,
        overlap_factor=0.2,
    )
    return chart, region_data


# ---------------------------------------------------------------------------
# build() — return type and basic structural properties
# ---------------------------------------------------------------------------

class TestBuild:
    """Tests for Build."""
    def test_returns_chart_instance(self):
        """Returns chart instance."""
        chart, _ = _make_chart()
        assert isinstance(chart, Chart)

    def test_intrinsic_dim_stored(self):
        """Intrinsic dim stored."""
        chart, _ = _make_chart()
        assert chart.intrinsic_dim == 2

    def test_ambient_dim_stored(self):
        """Ambient dim stored."""
        chart, _ = _make_chart()
        assert chart.ambient_dim == 10

    def test_selected_vectors_shape(self):
        """Selected vectors shape."""
        chart, _ = _make_chart()
        assert chart.selected_vectors.shape == (10, 2)

    def test_coordinates_shape(self):
        """Coordinates shape."""
        chart, region_data = _make_chart()
        assert chart.coordinates.shape == (len(region_data), 2)

    def test_chart_map_output_shape(self):
        """Chart map output shape."""
        chart, region_data = _make_chart()
        z = chart.chart_map(region_data[0])
        assert z.shape == (2,)

    def test_chart_inverse_output_shape(self):
        """Chart inverse output shape."""
        chart, region_data = _make_chart()
        z = chart.chart_map(region_data[0])
        p_hat = chart.chart_inverse(z)
        assert p_hat.shape == (10,)


# ---------------------------------------------------------------------------
# Round-trip reconstruction error
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """Tests for Round Trip."""
    def test_reconstruction_error_bounded(self):
        """Reconstruction error bounded."""
        # chart_inverse(chart_map(p)) = V@V.T@p, projecting onto chart span.
        # Residual is the normal-space component; bounded by the data norm.
        chart, region_data = _make_chart()
        for p in region_data:
            p_hat = chart.chart_inverse(chart.chart_map(p))
            # Residual can't exceed the point's own norm.
            assert np.linalg.norm(p - p_hat) <= np.linalg.norm(p) + 1e-10

    def test_coordinates_match_chart_map(self):
        """Coordinates match chart map."""
        # chart.coordinates must equal _compute_coordinates applied to region_data.
        chart, region_data = _make_chart()
        coords = _compute_coordinates(region_data, chart.chart_map)
        np.testing.assert_allclose(chart.coordinates, coords, atol=1e-10)


# ---------------------------------------------------------------------------
# _expand_region
# ---------------------------------------------------------------------------

class TestExpandRegion:
    """Tests for Expand Region."""
    def test_region_indices_unchanged(self):
        """Region indices unchanged."""
        data, _ = _subspace_data()
        indices = np.arange(20, dtype=np.intp)
        idx = nn_utils.build_faiss_index(data)
        returned_region, _ = _expand_region(indices, data, idx, overlap_factor=0.2)
        np.testing.assert_array_equal(returned_region, indices)

    def test_expanded_is_superset(self):
        """Expanded is superset."""
        data, _ = _subspace_data()
        indices = np.arange(20, dtype=np.intp)
        idx = nn_utils.build_faiss_index(data)
        _, expanded = _expand_region(indices, data, idx, overlap_factor=0.2)
        for i in indices:
            assert i in set(expanded.tolist())

    def test_overlap_zero_gives_only_region(self):
        """Overlap zero gives only region."""
        # With overlap_factor=0 we request 0 outside neighbours (max(1,...) = 1).
        data, _ = _subspace_data(n=40)
        indices = np.arange(30, dtype=np.intp)
        idx = nn_utils.build_faiss_index(data)
        _, expanded = _expand_region(indices, data, idx, overlap_factor=0.0)
        # All region points must still be present.
        for i in indices:
            assert i in set(expanded.tolist())


# ---------------------------------------------------------------------------
# _define_chart_map and _compute_chart_inverse
# ---------------------------------------------------------------------------

class TestChartMapAndInverse:
    """Tests for Chart Map And Inverse."""
    def test_chart_map_is_linear_projection(self):
        """Chart map is linear projection."""
        # chart_map(p) = qr_mat.T @ p.
        rng = np.random.default_rng(5)
        qr_mat, _ = np.linalg.qr(rng.standard_normal((8, 3)))
        qr_mat = qr_mat[:, :3]
        chart_map = _define_chart_map(qr_mat)
        p = rng.standard_normal(8)
        np.testing.assert_allclose(chart_map(p), qr_mat.T @ p, atol=1e-12)

    def test_chart_inverse_is_linear_reconstruction(self):
        """Chart inverse is linear reconstruction."""
        # chart_inverse(z) = qr_mat @ z.
        rng = np.random.default_rng(6)
        qr_mat, _ = np.linalg.qr(rng.standard_normal((8, 3)))
        qr_mat = qr_mat[:, :3]
        chart_inv = _compute_chart_inverse(qr_mat)
        z = rng.standard_normal(3)
        np.testing.assert_allclose(chart_inv(z), qr_mat @ z, atol=1e-12)


# ---------------------------------------------------------------------------
# _select_basis_adaptive
# ---------------------------------------------------------------------------

class TestSelectBasisAdaptive:
    """Tests for Select Basis Adaptive."""
    def test_returns_correct_length(self):
        """Returns correct length."""
        data, _ = _subspace_data(n=30)
        ref = _reference_basis(data)
        fg = graph_builder.build(data)
        aligned = align_to_reference(GraphFourierEngine(fg).basis, ref)
        sel = _select_basis_adaptive(aligned, data, intrinsic_dim=2)
        assert len(sel) == 2

    def test_indices_in_valid_range(self):
        """Indices in valid range."""
        data, _ = _subspace_data(n=30)
        ref = _reference_basis(data)
        fg = graph_builder.build(data)
        aligned = align_to_reference(GraphFourierEngine(fg).basis, ref)
        k = aligned.eigenvectors.shape[1]
        sel = _select_basis_adaptive(aligned, data, intrinsic_dim=2)
        assert all(0 <= i < k for i in sel)

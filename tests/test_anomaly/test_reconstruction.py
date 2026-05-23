"""Tests for anomaly/reconstruction.py.

Fixture: 2-D linear submanifold in 5-D built via the atlas pipeline.
The module-scoped fixture is built once and shared across all tests.
"""
from __future__ import annotations

import numpy as np
import pytest
from anomaly.reconstruction import (
    _batch_reconstruct,
    _decode,
    _encode,
    _find_chart,
    reconstruct,
)


# ---------------------------------------------------------------------------
# reconstruct() — shape and consistency
# ---------------------------------------------------------------------------

class TestReconstruct:
    """Tests for Reconstruct."""
    def test_returns_two_arrays(self, manifold_fixture):
        """Returns two arrays."""
        mf, data = manifold_fixture
        result = reconstruct(data, mf)
        assert len(result) == 2

    def test_reconstructed_shape(self, manifold_fixture):
        """Reconstructed shape."""
        mf, data = manifold_fixture
        reconstructed, _ = reconstruct(data, mf)
        assert reconstructed.shape == data.shape

    def test_residuals_shape(self, manifold_fixture):
        """Residuals shape."""
        mf, data = manifold_fixture
        _, residuals = reconstruct(data, mf)
        assert residuals.shape == data.shape

    def test_residuals_equal_data_minus_reconstructed(self, manifold_fixture):
        """Residuals equal data minus reconstructed."""
        mf, data = manifold_fixture
        reconstructed, residuals = reconstruct(data, mf)
        np.testing.assert_allclose(residuals, data - reconstructed, atol=1e-12)

    def test_reconstructed_lies_in_chart_span(self, manifold_fixture):
        """Reconstructed lies in chart span."""
        # p̂ = basis_vecs @ basis_vecs.T @ p is in span(basis_vecs);
        # the residual has zero projection onto basis_vecs.
        mf, data = manifold_fixture
        _, residuals = reconstruct(data, mf)
        for i in range(len(data)):
            ci = int(mf.atlas.primary_assignments[i])
            basis_vecs = mf.atlas.charts[ci].basis.selected_vectors
            # residual should be orthogonal to basis_vecs's columns
            np.testing.assert_allclose(
                basis_vecs.T @ residuals[i], np.zeros(basis_vecs.shape[1]), atol=1e-10
            )


# ---------------------------------------------------------------------------
# _find_chart
# ---------------------------------------------------------------------------

def test_returns_chart_at_assigned_index(manifold_fixture):
    """Returns chart at assigned index."""
    mf, _ = manifold_fixture
    for i in [0, 5, 10]:
        expected_ci = int(mf.atlas.primary_assignments[i])
        chart = _find_chart(i, mf)
        assert chart is mf.atlas.charts[expected_ci]


# ---------------------------------------------------------------------------
# _encode and _decode
# ---------------------------------------------------------------------------

class TestEncodeAndDecode:
    """Tests for Encode And Decode."""
    def test_encode_applies_chart_map(self, manifold_fixture):
        """Encode applies chart map."""
        mf, data = manifold_fixture
        chart = mf.atlas.charts[0]
        p = data[0]
        np.testing.assert_allclose(_encode(p, chart), chart.chart_map(p), atol=1e-12)

    def test_decode_applies_chart_inverse(self, manifold_fixture):
        """Decode applies chart inverse."""
        mf, data = manifold_fixture
        chart = mf.atlas.charts[0]
        z = chart.chart_map(data[0])
        np.testing.assert_allclose(_decode(z, chart), chart.chart_inverse(z), atol=1e-12)

    def test_encode_output_shape(self, manifold_fixture):
        """Encode output shape."""
        mf, data = manifold_fixture
        chart = mf.atlas.charts[0]
        z = _encode(data[0], chart)
        assert z.shape == (chart.intrinsic_dim,)

    def test_decode_output_shape(self, manifold_fixture):
        """Decode output shape."""
        mf, _ = manifold_fixture
        chart = mf.atlas.charts[0]
        z = np.zeros(chart.intrinsic_dim)
        p_hat = _decode(z, chart)
        assert p_hat.shape == (chart.ambient_dim,)


# ---------------------------------------------------------------------------
# _batch_reconstruct
# ---------------------------------------------------------------------------

class TestBatchReconstruct:
    """Tests for Batch Reconstruct."""
    def test_same_result_as_reconstruct(self, manifold_fixture):
        """Same result as reconstruct."""
        mf, data = manifold_fixture
        r1, res1 = reconstruct(data, mf)
        r2, res2 = _batch_reconstruct(data, mf, batch_size=5000)
        np.testing.assert_allclose(r1, r2, atol=1e-12)
        np.testing.assert_allclose(res1, res2, atol=1e-12)

    def test_small_batch_gives_same_result(self, manifold_fixture):
        """Small batch gives same result."""
        mf, data = manifold_fixture
        r_big, _ = _batch_reconstruct(data, mf, batch_size=10_000)
        r_small, _ = _batch_reconstruct(data, mf, batch_size=10)
        np.testing.assert_allclose(r_big, r_small, atol=1e-12)

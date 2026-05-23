"""Tests for anomaly/residual.py."""
from __future__ import annotations

import numpy as np
import pytest

from anomaly.reconstruction import reconstruct
from anomaly.residual import (
    ResidualData,
    _decompose_normal_to_bands,
    _normal_residual,
    compute,
)


# ---------------------------------------------------------------------------
# Module-scoped manifold fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def manifold_fixture():
    """Manifold fixture."""
    from common.types import FourierType, StructureType
    from fourier import analyze_fourier
    from manifold import build_manifold
    from structure.report import StructureReport
    import structure.graph_builder as gb

    rng = np.random.default_rng(1)
    d, intrinsic, n = 5, 2, 100
    orth_mat, _ = np.linalg.qr(rng.standard_normal((d, intrinsic)))
    basis = orth_mat[:, :intrinsic]
    coords = rng.standard_normal((n, intrinsic))
    data = coords @ basis.T + rng.standard_normal((n, d)) * 0.02

    fg = gb.build(data)
    structure = StructureReport(
        type=StructureType.GRAPH, intrinsic_dim=intrinsic, graph=fg,
        ordering=None, periodicity=False,
        recommended_fourier=FourierType.GRAPH_FOURIER, confidence=1.0,
    )
    spectral = analyze_fourier(data, structure)
    config = {"n_charts": 3, "overlap_factor": 0.2, "max_retries": 3}
    mf = build_manifold(data, spectral, structure, config)
    return mf, data


# ---------------------------------------------------------------------------
# compute() — return type and structure
# ---------------------------------------------------------------------------

class TestCompute:
    """Tests for Compute."""
    def test_returns_residual_data(self, manifold_fixture):
        """Returns residual data."""
        mf, data = manifold_fixture
        _, raw_residuals = reconstruct(data, mf)
        result = compute(raw_residuals, data, mf)
        assert isinstance(result, ResidualData)

    def test_total_residuals_stored(self, manifold_fixture):
        """Total residuals stored."""
        mf, data = manifold_fixture
        _, raw_residuals = reconstruct(data, mf)
        result = compute(raw_residuals, data, mf)
        np.testing.assert_array_equal(result.total_residuals, raw_residuals)

    def test_normal_residuals_shape(self, manifold_fixture):
        """Normal residuals shape."""
        mf, data = manifold_fixture
        _, raw_residuals = reconstruct(data, mf)
        result = compute(raw_residuals, data, mf)
        assert result.normal_residuals.shape == data.shape

    def test_per_band_keys_match_spectral_bands(self, manifold_fixture):
        """Per band keys match spectral bands."""
        mf, data = manifold_fixture
        _, raw_residuals = reconstruct(data, mf)
        result = compute(raw_residuals, data, mf)
        expected_keys = set(range(len(mf.spectral.bands)))
        assert set(result.per_band_norms.keys()) == expected_keys

    def test_per_band_norms_shape(self, manifold_fixture):
        """Per band norms shape."""
        mf, data = manifold_fixture
        _, raw_residuals = reconstruct(data, mf)
        result = compute(raw_residuals, data, mf)
        for norms in result.per_band_norms.values():
            assert norms.shape == (len(data),)

    def test_per_band_norms_non_negative(self, manifold_fixture):
        """Per band norms non negative."""
        mf, data = manifold_fixture
        _, raw_residuals = reconstruct(data, mf)
        result = compute(raw_residuals, data, mf)
        for norms in result.per_band_norms.values():
            assert np.all(norms >= 0.0)


# ---------------------------------------------------------------------------
# _normal_residual
# ---------------------------------------------------------------------------

class TestNormalResidual:
    """Tests for Normal Residual."""
    def test_output_shape(self, manifold_fixture):
        """Output shape."""
        mf, data = manifold_fixture
        chart = mf.atlas.charts[0]
        r = np.ones(data.shape[1])
        result = _normal_residual(r, data[0], chart)
        assert result.shape == (data.shape[1],)

    def test_tangent_vector_gives_zero_normal(self, manifold_fixture):
        """Tangent vector gives zero normal."""
        # A pure tangent vector has no normal component.
        mf, data = manifold_fixture
        chart = mf.atlas.charts[0]
        tangent_vec = chart.selected_vectors[:, 0]   # column of V
        result = _normal_residual(tangent_vec, data[0], chart)
        np.testing.assert_allclose(np.linalg.norm(result), 0.0, atol=1e-10)

    def test_projection_is_idempotent(self, manifold_fixture):
        """Projection is idempotent."""
        # Projecting a normal residual again gives the same result.
        mf, data = manifold_fixture
        chart = mf.atlas.charts[0]
        r = np.random.default_rng(5).standard_normal(data.shape[1])
        r_n = _normal_residual(r, data[0], chart)
        r_nn = _normal_residual(r_n, data[0], chart)
        np.testing.assert_allclose(r_n, r_nn, atol=1e-10)


# ---------------------------------------------------------------------------
# _decompose_normal_to_bands
# ---------------------------------------------------------------------------

class TestDecomposeNormalToBands:
    """Tests for Decompose Normal To Bands."""
    def test_returns_dict_with_band_keys(self, manifold_fixture):
        """Returns dict with band keys."""
        mf, data = manifold_fixture
        residuals = np.random.default_rng(7).standard_normal(data.shape)
        result = _decompose_normal_to_bands(
            residuals, mf.spectral.bands, mf.spectral.basis
        )
        assert set(result.keys()) == set(range(len(mf.spectral.bands)))

    def test_each_entry_has_n_rows(self, manifold_fixture):
        """Each entry has n rows."""
        mf, data = manifold_fixture
        residuals = np.random.default_rng(8).standard_normal(data.shape)
        result = _decompose_normal_to_bands(
            residuals, mf.spectral.bands, mf.spectral.basis
        )
        for norms in result.values():
            assert len(norms) == len(data)

    def test_norms_non_negative(self, manifold_fixture):
        """Norms non negative."""
        mf, data = manifold_fixture
        residuals = np.random.default_rng(9).standard_normal(data.shape)
        result = _decompose_normal_to_bands(
            residuals, mf.spectral.bands, mf.spectral.basis
        )
        for norms in result.values():
            assert np.all(norms >= 0.0)

    def test_zero_residuals_give_zero_norms(self, manifold_fixture):
        """Zero residuals give zero norms."""
        mf, data = manifold_fixture
        zeros = np.zeros_like(data)
        result = _decompose_normal_to_bands(
            zeros, mf.spectral.bands, mf.spectral.basis
        )
        for norms in result.values():
            np.testing.assert_allclose(norms, 0.0, atol=1e-12)

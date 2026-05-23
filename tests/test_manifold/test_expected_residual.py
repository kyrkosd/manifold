"""Tests for manifold/expected_residual.py.

Fixture: a 2-D linear subspace in 5-D ambient space.
On-manifold points: small normal residuals (only noise).
Planted anomalies: displaced off-manifold; larger normal residuals.
"""
from __future__ import annotations

import numpy as np
import pytest

from common import nn_utils
from common.types import EigenBasis, FrequencyBand, FourierType, StructureType
from fourier import SpectralData, analyze_fourier
from manifold.atlas import build as build_atlas
from manifold.chart import build as build_chart, _define_chart_map, _compute_chart_inverse
from manifold.eigenvector_alignment import AlignedBasis
from manifold.expected_residual import (
    ExpectedResidualDistribution,
    _decompose_to_bands,
    _encode_decode_residuals,
    _fit_distribution,
    _project_residuals_to_normal,
    compute,
)
from structure.report import StructureReport
import structure.graph_builder as graph_builder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_manifold_data(n: int = 80, d: int = 5, intrinsic: int = 2,
                        noise: float = 0.02, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (data, basis) for n points near a 2-D subspace in d-D."""
    rng = np.random.default_rng(seed)
    orth_mat, _ = np.linalg.qr(rng.standard_normal((d, intrinsic)))
    basis = orth_mat[:, :intrinsic]
    coords = rng.standard_normal((n, intrinsic))
    data = coords @ basis.T + rng.standard_normal((n, d)) * noise
    return data, basis


def _make_structure(data: np.ndarray, intrinsic_dim: int = 2) -> StructureReport:
    fg = graph_builder.build(data)
    return StructureReport(
        type=StructureType.GRAPH,
        intrinsic_dim=intrinsic_dim,
        graph=fg,
        ordering=None,
        periodicity=False,
        recommended_fourier=FourierType.GRAPH_FOURIER,
        confidence=1.0,
    )


def _make_chart_for_data(data: np.ndarray, structure: StructureReport):
    """Build a PCA-based chart — guaranteed to pass injectivity for submanifold data.

    We bypass GFT-chart validation here because expected_residual tests are about
    the residual computation logic, not the chart-building pipeline.
    """
    from sklearn.decomposition import PCA
    from manifold.chart import Chart, _define_chart_map, _compute_chart_inverse, _compute_coordinates
    from manifold.eigenvector_alignment import AlignedBasis
    from common.types import AlignmentQuality

    k = structure.intrinsic_dim
    pca = PCA(n_components=k)
    pca.fit(data)
    basis_vecs = pca.components_.T   # (d, k) orthonormal columns

    chart_map = _define_chart_map(basis_vecs)
    chart_inv = _compute_chart_inverse(basis_vecs)
    coords = _compute_coordinates(data, chart_map)

    dummy_basis = AlignedBasis(
        eigenvectors=basis_vecs,
        eigenvalues=pca.explained_variance_,
        rotation_matrix=np.eye(k),
        alignment_score=1.0,
        quality=AlignmentQuality.EXCELLENT,
    )

    return Chart(
        region_indices=np.arange(len(data), dtype=np.intp),
        expanded_indices=np.arange(len(data), dtype=np.intp),
        local_basis=dummy_basis,
        selected_indices=list(range(k)),
        selected_vectors=basis_vecs,
        coordinates=coords,
        chart_map=chart_map,
        chart_inverse=chart_inv,
        intrinsic_dim=k,
        ambient_dim=data.shape[1],
    )


def _make_spectral(data: np.ndarray, structure: StructureReport) -> SpectralData:
    return analyze_fourier(data, structure)


# ---------------------------------------------------------------------------
# compute() — return type and structure
# ---------------------------------------------------------------------------

class TestComputeDistribution:
    def test_returns_expected_residual_distribution(self):
        data, _ = _make_manifold_data()
        structure = _make_structure(data)
        spectral = _make_spectral(data, structure)
        chart = _make_chart_for_data(data, structure)
        result = compute(chart, data, spectral.bands, spectral.basis)
        assert isinstance(result, ExpectedResidualDistribution)

    def test_n_samples_matches_data(self):
        data, _ = _make_manifold_data()
        structure = _make_structure(data)
        spectral = _make_spectral(data, structure)
        chart = _make_chart_for_data(data, structure)
        result = compute(chart, data, spectral.bands, spectral.basis)
        assert result.n_samples == len(data)

    def test_per_band_keys_match_bands(self):
        data, _ = _make_manifold_data()
        structure = _make_structure(data)
        spectral = _make_spectral(data, structure)
        chart = _make_chart_for_data(data, structure)
        result = compute(chart, data, spectral.bands, spectral.basis)
        expected_keys = set(range(len(spectral.bands)))
        assert set(result.per_band_mean.keys()) == expected_keys
        assert set(result.per_band_var.keys()) == expected_keys

    def test_means_are_non_negative(self):
        data, _ = _make_manifold_data()
        structure = _make_structure(data)
        spectral = _make_spectral(data, structure)
        chart = _make_chart_for_data(data, structure)
        result = compute(chart, data, spectral.bands, spectral.basis)
        for m in result.per_band_mean.values():
            assert m >= 0.0

    def test_variances_are_non_negative(self):
        data, _ = _make_manifold_data()
        structure = _make_structure(data)
        spectral = _make_spectral(data, structure)
        chart = _make_chart_for_data(data, structure)
        result = compute(chart, data, spectral.bands, spectral.basis)
        for v in result.per_band_var.values():
            assert v >= 0.0

    def test_empty_data_returns_zero_samples(self):
        data, _ = _make_manifold_data()
        structure = _make_structure(data)
        spectral = _make_spectral(data, structure)
        chart = _make_chart_for_data(data, structure)
        result = compute(chart, np.empty((0, data.shape[1])), spectral.bands, spectral.basis)
        assert result.n_samples == 0


# ---------------------------------------------------------------------------
# _encode_decode_residuals
# ---------------------------------------------------------------------------

class TestEncodeDecodeResiduals:
    def test_residuals_shape(self):
        data, _ = _make_manifold_data(n=40)
        structure = _make_structure(data)
        chart = _make_chart_for_data(data, structure)
        residuals = _encode_decode_residuals(data, chart)
        assert residuals.shape == data.shape

    def test_residual_is_p_minus_p_hat(self):
        # r[i] must equal data[i] - chart_inverse(chart_map(data[i])).
        data, _ = _make_manifold_data(n=10)
        structure = _make_structure(data)
        chart = _make_chart_for_data(data, structure)
        residuals = _encode_decode_residuals(data, chart)
        for i, p in enumerate(data):
            expected = p - chart.chart_inverse(chart.chart_map(p))
            np.testing.assert_allclose(residuals[i], expected, atol=1e-12)


# ---------------------------------------------------------------------------
# _decompose_to_bands
# ---------------------------------------------------------------------------

class TestDecomposeToBands:
    def test_returns_dict_with_band_keys(self):
        data, _ = _make_manifold_data(n=40)
        structure = _make_structure(data)
        spectral = _make_spectral(data, structure)
        chart = _make_chart_for_data(data, structure)
        residuals = _encode_decode_residuals(data, chart)
        band_norms = _decompose_to_bands(residuals, spectral.bands, spectral.basis)
        assert set(band_norms.keys()) == set(range(len(spectral.bands)))

    def test_norms_non_negative(self):
        data, _ = _make_manifold_data(n=40)
        structure = _make_structure(data)
        spectral = _make_spectral(data, structure)
        chart = _make_chart_for_data(data, structure)
        residuals = _encode_decode_residuals(data, chart)
        band_norms = _decompose_to_bands(residuals, spectral.bands, spectral.basis)
        for norms in band_norms.values():
            assert np.all(norms >= 0.0)

    def test_each_band_norm_array_has_n_rows(self):
        data, _ = _make_manifold_data(n=40)
        structure = _make_structure(data)
        spectral = _make_spectral(data, structure)
        chart = _make_chart_for_data(data, structure)
        residuals = _encode_decode_residuals(data, chart)
        band_norms = _decompose_to_bands(residuals, spectral.bands, spectral.basis)
        for norms in band_norms.values():
            assert len(norms) == len(data)


# ---------------------------------------------------------------------------
# _fit_distribution
# ---------------------------------------------------------------------------

class TestFitDistribution:
    def test_returns_two_floats(self):
        norms = np.array([0.1, 0.2, 0.15, 0.3])
        mean, var = _fit_distribution(norms)
        assert isinstance(mean, float) and isinstance(var, float)

    def test_mean_matches_numpy(self):
        norms = np.array([1.0, 2.0, 3.0, 4.0])
        mean, _ = _fit_distribution(norms)
        assert mean == pytest.approx(float(np.mean(norms)))

    def test_var_matches_numpy(self):
        norms = np.array([1.0, 2.0, 3.0, 4.0])
        _, var = _fit_distribution(norms)
        assert var == pytest.approx(float(np.var(norms)))

    def test_empty_gives_zeros(self):
        mean, var = _fit_distribution(np.array([]))
        assert mean == 0.0 and var == 0.0

    def test_constant_gives_zero_var(self):
        norms = np.full(5, 2.5)
        _, var = _fit_distribution(norms)
        assert var == pytest.approx(0.0, abs=1e-10)

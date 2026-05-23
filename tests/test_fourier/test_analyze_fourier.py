"""Integration tests for fourier.analyze_fourier() and SpectralData.

These tests exercise the end-to-end Phase 2 pipeline: Phase 1 graph +
GFT + power spectrum + band decomposition → SpectralData.
"""
from __future__ import annotations

import numpy as np
import pytest

from common.types import FourierType, StructureType
from fourier import SpectralData, analyze_fourier
from structure.graph_builder import build as _build_graph
from structure.report import StructureReport, build as _build_report


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _report(n: int = 40, d: int = 5, seed: int = 0) -> tuple[np.ndarray, StructureReport]:
    """Return (data, StructureReport) built from random data."""
    data = np.random.default_rng(seed).standard_normal((n, d))
    graph = _build_graph(data)
    report = _build_report(graph, dim_result=2)
    return data, report


# ---------------------------------------------------------------------------
# analyze_fourier — output type and shape contracts
# ---------------------------------------------------------------------------

class TestAnalyzeFourierOutput:
    """Tests for Analyze Fourier Output."""
    def test_returns_spectral_data(self):
        """Returns spectral data."""
        # End-to-end call must produce a SpectralData instance.
        data, report = _report()
        assert isinstance(analyze_fourier(data, report), SpectralData)

    def test_coefficients_shape_matches_input(self):
        """Coefficients shape matches input."""
        # GFT is shape-preserving: (n, d) in → (n, d) out.
        data, report = _report(n=30, d=5, seed=1)
        assert analyze_fourier(data, report).coefficients.shape == (30, 5)

    def test_bands_list_is_non_empty(self):
        """Bands list is non empty."""
        # At least one frequency band must be produced for any data.
        data, report = _report(seed=2)
        assert len(analyze_fourier(data, report).bands) > 0

    def test_power_keys_present(self):
        """Power keys present."""
        # Power-spectrum dict must contain all five expected keys.
        data, report = _report(seed=3)
        result = analyze_fourier(data, report)
        assert {"per_point", "mean", "cumulative", "dominant", "centroid"}.issubset(
            result.power
        )

    def test_band_coefficients_keys_match_bands(self):
        """Band coefficients keys match bands."""
        # band_coefficients keys must match the labels of the bands list.
        data, report = _report(seed=4)
        result = analyze_fourier(data, report)
        band_labels = {b.label for b in result.bands}
        assert set(result.band_coefficients.keys()) == band_labels


# ---------------------------------------------------------------------------
# analyze_fourier — error handling
# ---------------------------------------------------------------------------

class TestAnalyzeFourierErrors:
    """Tests for Analyze Fourier Errors."""
    def test_none_graph_raises_value_error(self):
        """None graph raises value error."""
        # Missing feature graph must raise ValueError with a descriptive message.
        bad_report = StructureReport(
            type=StructureType.LINEAR,
            intrinsic_dim=1,
            graph=None,
            ordering=None,
            periodicity=False,
            recommended_fourier=FourierType.STANDARD_FFT,
            confidence=1.0,
        )
        with pytest.raises(ValueError, match="graph is None"):
            analyze_fourier(np.ones((5, 3)), bad_report)

"""Tests for fourier/band_decomposition.py.

Band coverage is the primary correctness invariant: every spectral index in
[0, d-1] must belong to exactly one FrequencyBand with no gaps or overlaps.
"""
from __future__ import annotations

import numpy as np
import pytest

from common.types import EigenBasis, FrequencyBand
from fourier.band_decomposition import (
    _assign_coefficients,
    _compute_band_power,
    _create_bands_at_gaps,
    _create_uniform_bands,
    _find_spectral_gaps,
    _label_bands,
    _validate_coverage,
    decompose,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _basis(d: int = 6) -> EigenBasis:
    """Identity eigenbasis with linearly spaced eigenvalues."""
    return EigenBasis(eigenvectors=np.eye(d), eigenvalues=np.arange(d, dtype=float))


def _coeffs(n: int = 10, d: int = 6, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, d))


# ---------------------------------------------------------------------------
# _find_spectral_gaps and _create_bands_at_gaps
# ---------------------------------------------------------------------------

class TestSpectralGapsAndBoundaries:
    """Tests for Spectral Gaps And Boundaries."""
    def test_flat_spectrum_has_no_gaps(self):
        # All ratios equal 1; no ratio exceeds 3× median → empty gap list.
        assert _find_spectral_gaps(np.ones(6)) == []

    def test_large_upward_jump_detected(self):
        # mean_power[2] is 100× larger than neighbours → gap at index 1.
        mean = np.array([1.0, 1.0, 100.0, 1.0, 1.0])
        assert 1 in _find_spectral_gaps(mean)

    def test_single_component_no_gaps(self):
        # No consecutive pairs → no gaps possible.
        assert _find_spectral_gaps(np.array([5.0])) == []

    def test_bands_at_single_gap(self):
        # Gap at index 2 with d=6 → two bands: (0,2) and (3,5).
        assert _create_bands_at_gaps([2], d=6) == [(0, 2), (3, 5)]

    def test_bands_at_gaps_cover_all_indices(self):
        # Two gaps divide d=9 into three bands; all indices must be covered.
        bounds = _create_bands_at_gaps([2, 5], d=9)
        covered = {i for s, e in bounds for i in range(s, e + 1)}
        assert covered == set(range(9))


# ---------------------------------------------------------------------------
# _create_uniform_bands and band helpers
# ---------------------------------------------------------------------------

class TestBandHelpers:
    """Tests for Band Helpers."""
    def test_uniform_bands_cover_all_indices(self):
        # Three uniform bands over d=6 must span {0, …, 5} without gaps.
        bounds = _create_uniform_bands(d=6, n_bands=3)
        covered = {i for s, e in bounds for i in range(s, e + 1)}
        assert covered == set(range(6))

    def test_assign_coefficients_shape(self):
        # Slice columns [1, 3] inclusive from (10, 6) → (10, 3).
        assert _assign_coefficients(_coeffs(), 1, 3).shape == (10, 3)

    def test_band_power_non_negative(self):
        # Sum of squares is always ≥ 0.
        assert _compute_band_power(_coeffs(), 0, 5) >= 0

    def test_band_power_known_value(self):
        # Single row [0, 3, 0]; band [1,1] contains column 1 → power = 9.
        coeffs = np.array([[0.0, 3.0, 0.0]])
        assert _compute_band_power(coeffs, 1, 1) == pytest.approx(9.0)

    def test_validate_coverage_valid(self):
        # Three consecutive non-overlapping bands covering [0, 5] → True.
        bands = [
            FrequencyBand(0, 1, "dc", 0.5),
            FrequencyBand(2, 3, "mid", 0.3),
            FrequencyBand(4, 5, "noise", 0.2),
        ]
        assert _validate_coverage(bands, d=6) is True

    def test_validate_coverage_gap_fails(self):
        # Index 2 is missing between the two bands → False.
        bands = [FrequencyBand(0, 1, "dc", 0.5), FrequencyBand(3, 5, "noise", 0.5)]
        assert _validate_coverage(bands, d=6) is False


# ---------------------------------------------------------------------------
# _label_bands
# ---------------------------------------------------------------------------

class TestLabelBands:
    """Tests for Label Bands."""
    def test_one_band_labelled_all(self):
        assert _label_bands([(0, 5)]) == ["all"]

    def test_two_bands_dc_noise(self):
        assert _label_bands([(0, 2), (3, 5)]) == ["dc", "noise"]

    def test_three_bands_dc_mid_noise(self):
        assert _label_bands([(0, 1), (2, 3), (4, 5)]) == ["dc", "mid", "noise"]

    def test_five_bands_full_set(self):
        # Five bands get the full dc / low / mid / high / noise label set.
        labels = _label_bands([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])
        assert labels == ["dc", "low", "mid", "high", "noise"]


# ---------------------------------------------------------------------------
# decompose() — integration
# ---------------------------------------------------------------------------

class TestDecompose:
    """Tests for Decompose."""
    def test_returns_frequency_bands(self):
        # decompose must produce at least one FrequencyBand object.
        mean = (_coeffs() ** 2).mean(axis=0)
        bands = decompose(_coeffs(), _basis(), mean)
        assert len(bands) > 0 and all(isinstance(b, FrequencyBand) for b in bands)

    def test_coverage_complete(self):
        # All d spectral components must be assigned to exactly one band.
        mean = (_coeffs() ** 2).mean(axis=0)
        bands = decompose(_coeffs(), _basis(), mean)
        assert _validate_coverage(bands, d=6)

    def test_band_power_non_negative(self):
        # Power is a sum of squares and must always be ≥ 0.
        mean = (_coeffs(seed=1) ** 2).mean(axis=0)
        assert all(b.power >= 0 for b in decompose(_coeffs(seed=1), _basis(), mean))
